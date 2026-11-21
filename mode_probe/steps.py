"""How many PnP-CM steps are needed, and which schedule is best?

Uses a re-implementation of the repo's loop that takes an explicit list of steps, each with its own
(sigma_t injected, sigma_cm told to the CM, rho penalty, mu momentum).  With the shipped per-step values it
reproduces the repo's sampler (checked in `curve`).

  curve  <task>              quality of the estimate after every step, shipped schedule, white vs shaped noise
  subsets <task>             every ordered subset of the shipped steps (which steps can be dropped?)
  search <task> <T> <noise>  evolution-strategy search of a T-step schedule on 8 calibration images
                             (sigma_t, sigma_cm per step + rho endpoints + mu_0), then scored on the 10 test
                             images with 3 sampler seeds.  noise = white | shaped
"shaped" = same-budget frequency-shaped noise (see shaped_noise.py); steps with sigma_t < 0.05 stay white.
Objective of the search: min(PSNR gain, LPIPS gain / 0.03) over the shipped 4-step schedule on calibration.

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/steps.py <mode> <task> ...
"""
import os, sys, json, glob, itertools
import numpy as np, torch
import torch.optim as optim
from PIL import Image

MODE, TASK = sys.argv[1], sys.argv[2]
EXTRA = sys.argv[3:]
sys.argv = ["random_steps.py", TASK, "--lib"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_steps as R

S, A, dev, T0, net = R.S, R.A, R.dev, R.T, R.model
LIN, PR, NB = TASK in R.LINEAR, TASK == "phase_retrieval", 64
SHP = (1, 3, 256, 256)
out = os.path.abspath("../mode_probe/random_steps/steps")
os.makedirs(out, exist_ok=True)

_fx = torch.fft.fftfreq(256, device=dev)
_rad = (_fx[:, None] ** 2 + _fx[None, :] ** 2).sqrt().flatten()
FB = (torch.argsort(torch.argsort(_rad)) * NB // _rad.numel()).long()
FCNT = torch.bincount(FB, minlength=NB).float()
fft_band = lambda p: torch.bincount(FB, weights=p.mean(dim=(0, 1)).flatten(), minlength=NB) / FCNT
softplus = torch.nn.Softplus()


def make_steps(sig_t, sig_cm, rho0, rho1, mu0):
    pen = softplus(torch.linspace(rho0, rho1, len(sig_t)))
    return [dict(st=float(np.clip(a, 0.002, 80)), scm=float(np.clip(b, 0.002, 80)), rho=float(pen[k]), mu=mu0 / (k + 1))
            for k, (a, b) in enumerate(zip(sig_t, sig_cm))]


def shipped_steps():
    al = [S.compute_alpha(R.betas, torch.tensor([R.hp["iN"] + 1], device=dev))]
    for _ in range(T0 - 1):
        al.append(S.get_next_alpha(al[-1], R.hp["gamma"]).reshape(1))
    st = [float(torch.sqrt(1 - a)) for a in al]
    return make_steps(st, [s * (1 + d) for s, d in zip(st, R.ship_deltas)], R.rhos[0], R.rhos[1], R.hp["mu_0"])


def cm(z, sigma):
    s = torch.tensor(sigma, device=dev)
    c_skip, c_out, c_in = S.get_scalings_for_boundary_condition(s)
    return (c_out * net(c_in * z, (250 * torch.log(s + 1e-44)).reshape(1)) + c_skip * z).clamp(-1, 1)


def datafit(zt, y, target, rho):
    """Inner solve of the nonlinear tasks, copied from the repo (including that it never zeroes gradients)."""
    o = R.opt
    with torch.enable_grad():
        z = zt.clone().requires_grad_(True)
        opt = getattr(optim, o["opt_type"])([z], lr=float(o["opt_lr"]))
        prev = 0.0
        for _ in range(o["opt_num_iter"]):
            res = A.A(z) - y
            diff = z - target
            loss = 0.5 * (res * res).sum() + 0.5 * rho * torch.sum(diff * diff).abs()
            loss.backward()
            opt.step()
            if loss.item() > prev:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * o["opt_decay_rate"]
            prev = loss.item()
    return z.detach()


SIG_Y, DIAG = 0.1, None  # measurement noise std on [-1,1] images; set DIAG = [] to log (step, residual ratio, rho)


def disc_rho(ATy, v, y, tau):
    """Discrepancy principle: the penalty for which the data-fit result leaves a residual equal to the
    expected measurement noise, ||A z - y||^2 = tau * m * sigma_y^2 (no fitting of the noise itself)."""
    tgt = tau * y.numel() * SIG_Y ** 2
    res = lambda r: float((A.A(A.AtA_add_eta_inv(ATy + r * v, eta=r).reshape(SHP)).reshape(y.shape) - y).pow(2).sum())
    lo, hi = 1e-4, 1e4
    if res(hi) <= tgt:
        return hi
    if res(lo) >= tgt:
        return lo
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        lo, hi = (lo, mid) if res(mid) > tgt else (mid, hi)
    return (lo * hi) ** 0.5


@torch.no_grad()
def pnp(y, xinit, steps, noise, seed, x0=None, tables=None, rec=None):
    torch.manual_seed(seed)
    x = xinit.clone() if LIN else torch.zeros(SHP, device=dev)
    u = torch.zeros(SHP, device=dev); xm = x.clone(); um = u.clone()
    if LIN:
        ATy = A.At(y).view(SHP)
    else:
        zt = S.phase_retrieval_initialization(y, 4.0) if PR else y.clone()
    specs = []
    for k, s in enumerate(steps):
        x1, u1 = x, u
        if s.get("prior"):  # no measurement update in this step: the CM refines the previous estimate directly
            z = x
        elif LIN:
            rho = disc_rho(ATy, xm - um, y, s.get("tau", 1.0)) if s.get("disc") else s["rho"]
            z = A.AtA_add_eta_inv(ATy + rho * (xm - um), eta=rho).reshape(SHP)
            if DIAG is not None:
                DIAG.append((k, float((A.A(z).reshape(y.shape) - y).pow(2).sum()) / (y.numel() * SIG_Y ** 2), rho))
        else:
            z = zt = datafit(zt, y, xm - um, s["rho"])
        r = z if s.get("prior") else z + um
        sp = None
        if x0 is not None:  # error spectrum of the iterate (calibration images only)
            ref = x0 * torch.sign((r * x0).sum(dim=(2, 3), keepdim=True)) if PR else x0
            sp = fft_band(torch.fft.fft2(r - ref, norm="ortho").abs() ** 2)
            specs.append(sp)
        def draw():
            w = torch.randn_like(xm)
            if noise == "white" or s["st"] < 0.05:
                return w * s["st"]
            P = tables[k] if tables is not None else sp
            g = (s["st"] ** 2 + P.mean() - P[FB].view(256, 256)).clamp(min=0).sqrt()
            return torch.fft.ifft2(torch.fft.fft2(w, norm="ortho") * g, norm="ortho").real

        if s.get("dc"):  # data-consistency only: no network call in this step
            x = z.clamp(-1, 1)
        elif s.get("K", 1) > 1:  # several noise draws in this step: average them, or keep the one that fits y best
            xs = [cm(r + draw(), s["scm"]) for _ in range(s["K"])]
            if s.get("pick"):
                x = xs[int(np.argmin([float((A.A(v).reshape(y.shape) - y).pow(2).sum()) for v in xs]))]
            else:
                x = torch.stack(xs).mean(0)
        else:
            x = cm(r + draw(), s["scm"])
        if s.get("prior"):
            xm = x
        else:
            u = um + (z - x)
            xm, um = x + s["mu"] * (x - x1), u + s["mu"] * (u - u1)
        if rec is not None:
            rec.append(x)
    return x, specs


FL = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3)), device=dev).view(8, 3, 1, 1)


def score(x, x0, final=True):
    if PR:
        x = R.W.adjust_color_by_corr(x, R.W.loss_fn_vgg, R.pr_ref) if final else FL[int(np.argmax([R.psnr(f * x, x0) for f in FL]))] * x
    return R.psnr(x, x0), float(R.lp(x, x0))


# ---- data
_f = sorted(glob.glob("/usr/scratch/akheirandish3/diffgeo/data/celebahq256/*.png"))
_im = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).float().permute(2, 0, 1) / 255 for p in _f]).to(dev)
_tt = torch.cat([(x0 + 1) / 2 for x0, _ in R.data])
_ok = np.where((torch.stack([((_im - t[None]) ** 2).mean(dim=(1, 2, 3)) for t in _tt], 1).min(1).values > 1e-3).cpu().numpy())[0]
cal = R.make_data([_im[i][None] for i in np.random.default_rng(0).permutation(_ok)[:8]], 7000)
with torch.no_grad():
    CAL_INIT = [R.x_init_of(y) if LIN else None for _, y in cal]
    TEST_INIT = [R.x_init_of(y) if LIN else None for _, y in R.data]


def cal_eval(steps, noise, want_tables=False):
    """Mean (PSNR, LPIPS) on the calibration images; shaped noise uses each image's own error spectrum."""
    sc, specs = [], []
    for i, (x0, y) in enumerate(cal):
        x, sp = pnp(y, CAL_INIT[i], steps, noise, 100 + i, x0=x0)
        sc.append(score(x, x0)); specs.append(torch.stack(sp))
    tab = list(torch.stack(specs).mean(0)) if want_tables else None
    return float(np.mean([s[0] for s in sc])), float(np.mean([s[1] for s in sc])), tab


def test_eval(steps, noise, tables, seeds=3, curve=False):
    res, per = [], []
    for sd in range(seeds):
        for i, (x0, y) in enumerate(R.data):
            rec = []
            x, _ = pnp(y, TEST_INIT[i], steps, noise, i + 1000 * sd, tables=tables, rec=rec)
            res.append(score(x, x0))
            if curve:
                per.append([score(xk, x0, final=False) for xk in rec])
    m = (float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res])))
    return (m, np.mean(np.array(per), 0)) if curve else m


SHIP = shipped_steps()
desc = lambda st: "sigma_t " + str([round(s["st"], 3) for s in st]) + " told " + str([round(s["scm"], 3) for s in st]) + \
    " rho " + str([round(s["rho"], 3) for s in st]) + " mu " + str([round(s["mu"], 3) for s in st])

if MODE == "curve":
    seeds = 1 if PR else 3
    print(f"\n######## {TASK}: estimate after each step, shipped schedule ({desc(SHIP)})")
    for noise in ("white", "shaped"):
        if noise == "white":
            tab = None
        elif PR:
            tab = [torch.tensor(s, device=dev, dtype=torch.float32) for s in
                   json.load(open(os.path.abspath("../mode_probe/random_steps/shaped/phase_retrieval_cal_shape.json")))["spec"]]
        else:
            tab = cal_eval(SHIP, "shaped", True)[2]
        m, per = test_eval(SHIP, noise, tab, seeds=seeds, curve=True)
        print(f"  {noise:6s} final PSNR {m[0]:.2f} LPIPS {m[1]:.3f}" + ("  (per-step PSNR below uses the best sign)" if PR else ""))
        print("     PSNR  after step: " + "  ".join(f"{k + 1}:{v:6.2f}" for k, v in enumerate(per[:, 0])))
        print("     LPIPS after step: " + "  ".join(f"{k + 1}:{v:6.3f}" for k, v in enumerate(per[:, 1])), flush=True)

elif MODE == "subsets":
    print(f"\n######## {TASK}: every ordered subset of the shipped steps, each step keeping its own shipped settings")
    print("steps kept   NFE | white noise PSNR / LPIPS | shaped noise PSNR / LPIPS")
    rows = []
    for n in range(1, T0 + 1):
        for sub in itertools.combinations(range(T0), n):
            st = [SHIP[k] for k in sub]
            w = test_eval(st, "white", None)
            s = test_eval(st, "shaped", cal_eval(st, "shaped", True)[2]) if any(q["st"] >= 0.05 for q in st) else w
            rows.append(dict(steps=[k + 1 for k in sub], white=w, shaped=s))
            print(f"{str([k + 1 for k in sub]):12s} {n:3d} | {w[0]:11.2f} / {w[1]:.3f}    | {s[0]:11.2f} / {s[1]:.3f}", flush=True)
    json.dump(rows, open(f"{out}/{TASK}_subsets.json", "w"))

elif MODE == "search":
    T, noise = int(EXTRA[0]), EXTRA[1]
    pop, gens = int(EXTRA[2]) if len(EXTRA) > 2 else 12, int(EXTRA[3]) if len(EXTRA) > 3 else 25
    rng = np.random.default_rng(10 * T + (1 if noise == "shaped" else 0))
    ref = cal_eval(SHIP, "white")[:2]
    LO, HI = np.log(0.002), np.log(80.0)

    def decode(v):
        return make_steps(np.exp(v[:T]), np.exp(v[T:2 * T]), float(v[2 * T]), float(v[2 * T + 1]), float(v[2 * T + 2]))

    hist = []

    def f(v):
        p, l, _ = cal_eval(decode(v), noise)
        s = min(p - ref[0], (ref[1] - l) / 0.03)
        s = s if np.isfinite(s) else -1e9
        hist.append(dict(v=v.tolist(), psnr=p, lpips=l, score=s))
        return s

    keep = {4: [0, 1, 2, 3], 3: [0, 1, 3], 2: [0, 1], 1: [0]}.get(T, list(range(T)))
    keep = [k for k in keep if k < T0][:T]
    v0 = np.array([np.log(SHIP[k]["st"]) for k in keep] + [np.log(SHIP[k]["scm"]) for k in keep] + [R.rhos[0], R.rhos[1], R.hp["mu_0"]])
    cands = [v0]
    for _ in range(pop - 1):
        st = np.sort(rng.uniform(np.log(0.01), np.log(1.0), T))[::-1]
        cands.append(np.concatenate([st, st + np.log1p(rng.uniform(0, 0.5, T)),
                                     [R.rhos[0] + rng.normal(0, 1.5), R.rhos[1] + rng.normal(0, 1.5), rng.uniform(0, 0.3)]]))
    elite = sorted([(f(v), v) for v in cands], key=lambda e: -e[0])[:4]
    step = 1.0
    for g in range(gens):
        kids = []
        for _ in range(pop):
            v = elite[rng.integers(len(elite))][1] + np.concatenate([rng.normal(0, 0.4 * step, 2 * T), rng.normal(0, 1.0 * step, 2), rng.normal(0, 0.08 * step, 1)])
            v[:2 * T] = np.clip(v[:2 * T], LO, HI); v[2 * T:2 * T + 2] = np.clip(v[2 * T:2 * T + 2], -10, 14); v[-1] = np.clip(v[-1], 0, 0.6)
            kids.append((f(v), v))
        elite = sorted(elite + kids, key=lambda e: -e[0])[:4]
        step = max(step * 0.9, 0.12)
        print(f"SEARCH {TASK} T={T} {noise} gen {g + 1}/{gens} best {elite[0][0]:+.3f}", flush=True)
    best = decode(elite[0][1])
    cp, cl, tab = cal_eval(best, noise, True)
    tp, tl = test_eval(best, noise, tab if noise == "shaped" else None)
    sp, sl = test_eval(SHIP, "white", None)
    json.dump(dict(task=TASK, T=T, noise=noise, steps=best, cal=(cp, cl), cal_ref=ref, test=(tp, tl), test_shipped=(sp, sl),
                   evals=len(hist)), open(f"{out}/{TASK}_search_T{T}_{noise}.json", "w"))
    print(f"FINAL {TASK} T={T} {noise}: cal {cp:.2f}/{cl:.3f} (shipped {ref[0]:.2f}/{ref[1]:.3f}) | test {tp:.2f}/{tl:.3f} "
          f"(shipped 4-step {sp:.2f}/{sl:.3f}) | {desc(best)}", flush=True)
