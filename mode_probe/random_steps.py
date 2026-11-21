"""How much do PnP-CM's hand-tuned noise steps matter?  Random-schedule experiment.

Everything except the noise schedule (rhos, mu_0, data-fit optimizer, measurements, sampler
seed) is held at the shipped values.  The repo's sampler code is used UNMODIFIED: explicit
(sigma_t, sigma_cm) lists are fed in by temporarily replacing the two schedule helpers
(compute_alpha / get_next_alpha) with a preset alpha list (alpha = 1 - sigma_t^2) and
passing deltas = sigma_cm / sigma_t - 1.

Conditions
  shipped        task YAML values, several sampler seeds (gives the seed noise floor)
  matched        shipped injected sigma, but CM told the true sigma (all deltas = 0)
  jitter         random draw inside the authors' own parameterization:
                 iN ~ U{50..300}, gamma ~ U[0.03, 0.3], delta_i ~ U[0.05, 0.8]
  rand_sorted    sigma_t log-uniform in [0.01, 1], sorted decreasing, delta_i ~ U[0, 0.5]
  rand_unsorted  same draws, order left random
  wide_sorted    sigma_t log-uniform in [0.01, 80], sorted decreasing, delta_i ~ U[0, 0.5]

Run from the PnP-CM repo root:
  CUDA_VISIBLE_DEVICES=g python ../mode_probe/random_steps.py <task> --worker i --nworkers n
  task in {deblur_gauss, deblur_nl, phase_retrieval};  --check verifies the patching trick.
"""
import os, sys, json, time, argparse, itertools
import numpy as np, torch, yaml, lpips
import torchvision.utils as tvu

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath("bkse"))

p = argparse.ArgumentParser()
p.add_argument("task")
p.add_argument("--worker", type=int, default=0)
p.add_argument("--nworkers", type=int, default=1)
p.add_argument("--check", action="store_true")
p.add_argument("--list", action="store_true")
p.add_argument("--lib", action="store_true", help="set up model/data/operator only (for import by calibrate.py)")
a = p.parse_args()
task = a.task
LINEAR = ("deblur_gauss", "sr_bicubic", "inpainting")
N_RAND = {"deblur_nl": 20, "phase_retrieval": 6}.get(task, 30)
N_SEED = {"phase_retrieval": 3}.get(task, 5)

ya = yaml.safe_load(open(f"task_specific_args/celeba_hq_{task}_sigma_y_0.05.yml"))
hp, opt = ya["hyperparameters"], ya.get("dc_optimizer", {})
T = hp["T_sampling"]
ship_deltas = [float(v) for v in hp["deltas"].split(",")]
rhos = [float(v) for v in hp["rhos"].split(",")]


def param_sigmas(iN, gamma, deltas):
    """sigma_t / sigma_cm that the repo's (iN, gamma, deltas) parameterization produces."""
    b = np.linspace(1e-4, 0.02, 1000)
    ac = np.concatenate([[1.0], np.cumprod(1 - b)])
    al, st = ac[iN + 2], []
    for _ in range(T):
        st.append(float(np.sqrt(1 - al)))
        al = min(al * (1 + gamma), 0.9999)
    return st, [s * (1 + d) for s, d in zip(st, deltas)]


def build_configs():
    cfgs = []
    st, sc = param_sigmas(hp["iN"], hp["gamma"], ship_deltas)
    for s in range(N_SEED):
        cfgs.append(dict(cond="shipped", mode="param", seed=s, iN=hp["iN"], gamma=hp["gamma"],
                         deltas=ship_deltas, sigma_t=st, sigma_cm=sc))
    cfgs.append(dict(cond="matched", mode="param", seed=0, iN=hp["iN"], gamma=hp["gamma"],
                     deltas=[0.0] * T, sigma_t=st, sigma_cm=st))
    # draws depend only on T, so the two 4-step tasks see the same random schedules
    rng = np.random.default_rng(1000 + T)
    for _ in range(30):
        iN, g = int(rng.integers(50, 301)), float(rng.uniform(0.03, 0.3))
        d = rng.uniform(0.05, 0.8, T).tolist()
        st, sc = param_sigmas(iN, g, d)
        cfgs.append(dict(cond="jitter", mode="param", seed=0, iN=iN, gamma=g, deltas=d, sigma_t=st, sigma_cm=sc))
    rng = np.random.default_rng(2000 + T)
    for _ in range(30):
        raw = np.exp(rng.uniform(np.log(0.01), np.log(1.0), T))
        d = rng.uniform(0.0, 0.5, T)
        srt = np.sort(raw)[::-1]
        cfgs.append(dict(cond="rand_sorted", mode="sigma", seed=0, sigma_t=srt.tolist(), sigma_cm=(srt * (1 + d)).tolist()))
        cfgs.append(dict(cond="rand_unsorted", mode="sigma", seed=0, sigma_t=raw.tolist(), sigma_cm=(raw * (1 + d)).tolist()))
    rng = np.random.default_rng(3000 + T)
    for _ in range(30):
        srt = np.sort(np.exp(rng.uniform(np.log(0.01), np.log(80.0), T)))[::-1]
        d = rng.uniform(0.0, 0.5, T)
        cfgs.append(dict(cond="wide_sorted", mode="sigma", seed=0, sigma_t=srt.tolist(), sigma_cm=(srt * (1 + d)).tolist()))
    keep, seen = [], {}
    for c in cfgs:
        seen[c["cond"]] = seen.get(c["cond"], 0) + 1
        if c["cond"] in ("shipped", "matched") or seen[c["cond"]] <= N_RAND:
            keep.append(c)
    for i, c in enumerate(keep):
        c["id"] = i
    return keep


configs = build_configs()
if a.list:
    for c in configs:
        print(c["id"], c["cond"], c["seed"], np.round(c["sigma_t"], 3), np.round(c["sigma_cm"], 3))
    sys.exit()

# ---------------------------------------------------------------- model / data / operator
from argparse import Namespace
from guided_diffusion.script_util import create_model
from guided_diffusion.diffusion_pnp_cm import get_beta_schedule
from datasets import get_dataset, data_transform

dev = torch.device("cuda:0")


def ns(d):
    return Namespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})


config = ns(yaml.safe_load(open("configs/celeba_hq_256.yml")))
config.device = dev
args = Namespace(exp="exp", deg=task, T_sampling=T, phase_ret_os=4.0, **opt)
betas = torch.from_numpy(get_beta_schedule(
    beta_schedule=config.diffusion.beta_schedule, beta_start=config.diffusion.beta_start,
    beta_end=config.diffusion.beta_end, num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps)).float().to(dev)

md = dict(vars(config.model)); md["image_size"] = config.data.image_size
model = create_model(**md)
model.load_state_dict(torch.load("exp/logs/" + ya["checkpoint"]["model_ckpt"], map_location=dev))
model.to(dev).eval()
lp = lpips.LPIPS(net="vgg").to(dev)

torch.manual_seed(1234); np.random.seed(1234)
x_init_of = lambda y: torch.zeros(1, 3, 256, 256, device=dev)
if task in LINEAR:
    import pnp_cm_scheme.natural_linear_scheme as S
    if task == "deblur_gauss":
        from functions.svd_operators import Deblurring
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / 10) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(dev)
        A = Deblurring(kernel / kernel.sum(), 3, 256, dev)
    elif task == "sr_bicubic":  # x4, same construction as wrappers/natural_linear_wrapper.py
        from functions.svd_operators import SRConv

        def _bic(x, a=-0.5):
            x = abs(x)
            return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1 if x <= 1 else (a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a if x < 2 else 0)

        _k = np.array([_bic((1 / 4) * (i - np.floor(16 / 2) + 0.5)) for i in range(16)])
        kernel = torch.from_numpy(_k / _k.sum()).float().to(dev)
        A = SRConv(kernel / kernel.sum(), 3, 256, dev, stride=4)
    else:  # random 70% inpainting, median-filled initialization as in the wrapper
        from functions.svd_operators import Inpainting
        from functions.median_inpainting import median_inpainting
        _mask = torch.from_numpy(np.load("exp/inpainting_masks/" + ya["degradation"]["inpainting_mask_path"])).to(dev).reshape(-1)
        _mr = torch.nonzero(_mask == 0).long().reshape(-1) * 3
        A = Inpainting(3, 256, torch.cat([_mr, _mr + 1, _mr + 2], dim=0), dev)
        x_init_of = lambda y: median_inpainting(A.A_pinv_add_eta(y).view(1, 3, 256, 256), _mask, 3, 256).view(1, 3, 256, 256)
else:
    import pnp_cm_scheme.natural_nonlinear_scheme as S
    import wrappers.natural_nonlinear_wrapper as W
    if task == "deblur_nl":
        from functions.measurements import NonlinearBlurOperator
        A = NonlinearBlurOperator("bkse/options/generate_blur/default.yml", dev)
        A.blur_model.eval()
    else:
        import scipy.io as sio
        from functions.measurements import PhaseRetrievalOperator
        A = PhaseRetrievalOperator(args.phase_ret_os, dev)
        pr_ref = torch.from_numpy(sio.loadmat("exp/phase_retrieval_ref_img/cm_out.mat")["cm_out"]).to(dev)

dataset = get_dataset(args, config)


def make_data(imgs, seed0):
    """imgs: list of [1,3,256,256] tensors in [0,1] -> list of (x0 in [-1,1], noisy measurement y)."""
    out = []
    with torch.no_grad():
        for i, im in enumerate(imgs):
            x0 = data_transform(config, im.to(dev))
            torch.manual_seed(seed0 + i)
            y = A.A(x0)
            out.append((x0, y + 2 * 0.05 * torch.randn_like(y)))  # sigma_y = 0.05 on [0,1] images
    return out


data = make_data([dataset[i][0][None] for i in range(len(dataset))], 5000)  # the 10 test images


def restore(cfg, y):
    def call(iN, gamma, deltas):
        if task in LINEAR:
            return S.pnp_cm_natural_linear_restoration(
                ATy=A.At(y).view(1, 3, 256, 256), x_init=x_init_of(y), model=model,
                A_funcs=A, betas=betas, iN=iN, gamma=gamma, deltas=deltas, rhos=rhos, mu_0=hp["mu_0"],
                T_sampling=T, config=config, classes=None)
        return S.pnp_cm_natural_nonlinear_restoration(
            y=y.detach(), model=model, A_funcs=A, betas=betas, iN=iN, gamma=gamma, deltas=deltas, rhos=rhos,
            mu_0=hp["mu_0"], classes=None, args=args, config=config)

    if cfg["mode"] == "param":
        return call(cfg["iN"], cfg["gamma"], cfg["deltas"])
    st, sc = np.array(cfg["sigma_t"]), np.array(cfg["sigma_cm"])
    it = iter([torch.tensor([1 - s ** 2], dtype=torch.float32, device=dev) for s in st])
    orig = (S.compute_alpha, S.get_next_alpha)
    S.compute_alpha = lambda b, t: next(it)
    S.get_next_alpha = lambda prev, g: next(it)
    try:
        return call(0, 0.0, (sc / st - 1).tolist())
    finally:
        S.compute_alpha, S.get_next_alpha = orig


def psnr(x, x0):
    return float(10 * torch.log10(1 / ((((x + 1) / 2).clamp(0, 1) - ((x0 + 1) / 2).clamp(0, 1)) ** 2).mean()))


FLIPS = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3)), device=dev).view(8, 3, 1, 1)


@torch.no_grad()
def run(cfg, save_dir=None, data=data):
    rows = []
    for i, (x0, y) in enumerate(data):
        torch.manual_seed(100003 * cfg["seed"] + i)
        x = restore(cfg, y)
        r = {}
        if task == "phase_retrieval":
            r["psnr_oracle_sign"] = max(psnr(f * x, x0) for f in FLIPS)
            x = W.adjust_color_by_corr(x, W.loss_fn_vgg, pr_ref)  # the repo's reference-face sign fix
        r["psnr"], r["lpips"] = psnr(x, x0), float(lp(x, x0))
        rows.append(r)
        if save_dir and i < 2:
            tvu.save_image(((x[0] + 1) / 2).clamp(0, 1), f"{save_dir}/c{cfg['id']:03d}_img{i}.png")
    return rows


if a.check:
    # patched explicit-sigma path must reproduce the unpatched shipped path
    c0 = configs[0]
    c1 = dict(c0, mode="sigma")
    x0, y = data[0]
    with torch.no_grad():
        torch.manual_seed(0); xa = restore(c0, y)
        torch.manual_seed(0); xb = restore(c1, y)
        torch.manual_seed(1); xc = restore(c0, y)
        torch.manual_seed(0); xd = restore(c0, y)
    print(f"identical rerun of param path: max diff {float((xa - xd).abs().max()):.2e}, "
          f"PSNR param {psnr(xa, x0):.3f} patched {psnr(xb, x0):.3f} rerun {psnr(xd, x0):.3f} other-seed {psnr(xc, x0):.3f}")
    print(f"CHECK {task}: max|param - patched| = {float((xa - xb).abs().max()):.2e}   "
          f"(different sampler seed for scale: {float((xa - xc).abs().max()):.2e})   PSNR {psnr(xa, x0):.2f}")
    sys.exit()

if a.lib:
    configs_all = configs
    configs = []
out = os.path.abspath(f"../mode_probe/random_steps/{task}")
os.makedirs(out, exist_ok=True)
for cfg in configs[a.worker::a.nworkers]:
    f = f"{out}/c{cfg['id']:03d}.json"
    if os.path.exists(f):
        continue
    t0 = time.time()
    cfg["rows"] = run(cfg, out)
    json.dump(cfg, open(f, "w"))
    print(f"DONE {task} c{cfg['id']:03d} {cfg['cond']:13s} PSNR {np.mean([r['psnr'] for r in cfg['rows']]):.2f} "
          f"LPIPS {np.mean([r['lpips'] for r in cfg['rows']]):.3f}  ({time.time() - t0:.0f}s)", flush=True)
