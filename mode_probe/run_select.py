"""PnP-CM phase retrieval with GT-free, reference-free early sign-mode selection.

Runs the unmodified repo pipeline (same seeds / RNG stream) but swaps in a
copy of the nonlinear sampler that can, after the data-fit solve of chosen
steps, score the 8 per-channel sign candidates with the consistency model and
continue from the winner.  The raw (un-fixed) output is saved per image so all
final-fix rules can be compared offline.

Run from the PnP-CM repo root:
  SEL_STEPS=<''|0|all> SEL_MODE=<x|z> python ../mode_probe/run_select.py <variant_name>
"""
import os, sys, itertools
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, ".")
variant = sys.argv[1]
SEL = os.environ.get("SEL_STEPS", "")
MODE = os.environ.get("SEL_MODE", "x")
SIG_SCORE = float(os.environ.get("SEL_SIGMA", "0.3"))
N_EPS = 2

# private exp dir (main.py wipes its image folder), sharing ckpt/data by symlink
exp = os.path.abspath(f"../mode_probe/exp_{variant}")
os.makedirs(exp, exist_ok=True)
for sub in ["logs", "datasets"]:
    if not os.path.exists(f"{exp}/{sub}"):
        os.symlink(os.path.abspath(f"exp/{sub}"), f"{exp}/{sub}")
sys.argv = ["main.py", "--deg=phase_retrieval", "--path_y=celeba_hq", "--sigma_y=0.05",
            "--config=celeba_hq_256.yml", "--save_y", f"--exp={exp}"]

import main as M
import wrappers.natural_nonlinear_wrapper as W
from pnp_cm_scheme.natural_nonlinear_scheme import (
    get_scalings_for_boundary_condition, append_dims, get_next_alpha, compute_alpha,
    phase_retrieval_initialization)

FLIPS = list(itertools.product([1.0, -1.0], repeat=3))
state = {"img": -1, "log": []}


def restoration(y, model, A_funcs, betas, iN, gamma, deltas, rhos, mu_0, classes, args, config,
                sigma_min=0.002, sigma_max=80.0):
    state["img"] += 1
    dev = y.device
    gsel = torch.Generator(device=dev).manual_seed(999 + state["img"])  # separate stream: base RNG untouched
    F = torch.tensor(FLIPS, device=dev).view(8, 3, 1, 1)
    softplus = nn.Softplus()
    penalty_param = softplus(torch.linspace(rhos[0], rhos[1], args.T_sampling)).to(dev)
    momentum = (torch.tensor([mu_0]).repeat(args.T_sampling) / torch.arange(1, args.T_sampling + 1)).to(dev)
    shp = (config.sampling.batch_size, config.data.channels, config.data.image_size, config.data.image_size)
    xt = torch.zeros(shp, device=dev); xt_moment = torch.zeros(shp, device=dev)
    ut = torch.zeros(shp, device=dev); ut_moment = torch.zeros(shp, device=dev)
    zt = phase_retrieval_initialization(y, args.phase_ret_os)

    t = (torch.ones(1) * (iN + 1)).to(dev)
    alphas = [compute_alpha(betas, t.long())]
    for _ in range(args.T_sampling - 1):
        alphas.append(get_next_alpha(alphas[-1], gamma).reshape(1))
    sel_steps = set(range(args.T_sampling)) if SEL == "all" else {int(s) for s in SEL.split(",") if s != ""}

    def cm(x, sigma):
        cs, co, ci = [append_dims(s, x.ndim) for s in get_scalings_for_boundary_condition(sigma)]
        tt = (1000 * 0.25 * torch.log(sigma + 1e-44)).reshape(-1).repeat(x.shape[0])
        return torch.clamp(co * model(ci * x, tt) + cs * x, -1.0, 1.0)

    def prior_score(c):
        s = torch.tensor(SIG_SCORE, device=dev)
        acc = torch.zeros(c.shape[0], device=dev)
        for _ in range(N_EPS):
            e = torch.randn(c[:1].shape, device=dev, generator=gsel)
            acc += ((cm(c + s * e, s) - c) ** 2).mean(dim=(1, 2, 3))
        return acc / N_EPS

    picks = []
    for iter_ind, at in enumerate(tqdm(alphas)):
        sigma_t_i = torch.sqrt(1 - at)
        sigma_cm_i = torch.clamp((1 + deltas[iter_ind]) * sigma_t_i, sigma_min, sigma_max)
        sigma_t_i = torch.clamp(sigma_t_i, sigma_min, sigma_max)
        xt_1, ut_1 = xt.clone(), ut.clone()
        loss_pre = torch.tensor(0.0, device=dev)

        with torch.enable_grad():
            ztmp = zt.clone().requires_grad_(True)
            optimizer = getattr(optim, args.opt_type)([ztmp], lr=float(args.opt_lr))
            for _ in range(args.opt_num_iter):
                res = A_funcs.A(ztmp) - y
                diff = ztmp - (xt_moment - ut_moment)
                loss = 0.5 * (res * res).sum() + 0.5 * penalty_param[iter_ind] * torch.sum(diff * diff).abs()
                loss.backward()
                optimizer.step()
                if loss.item() > loss_pre.item():
                    for g in optimizer.param_groups:
                        g["lr"] = g["lr"] * args.opt_decay_rate
                loss_pre = loss
            zt = ztmp.detach()
        del optimizer

        if iter_ind in sel_steps:
            # the 8 sign candidates fit y identically; let the prior choose
            if MODE == "z":
                sc = prior_score(F * zt)
            else:
                e = torch.randn(zt.shape, device=dev, generator=gsel)
                xc = cm(F * (zt + ut_moment) + e * sigma_t_i[0], sigma_cm_i)
                sc = prior_score(xc)
            k = int(sc.argmin())
            picks.append((iter_ind, k, [round(v, 5) for v in sc.tolist()]))
            f = F[k:k + 1]
            zt, xt, ut, xt_moment, ut_moment, xt_1, ut_1 = [f * v for v in (zt, xt, ut, xt_moment, ut_moment, xt_1, ut_1)]

        z_noisy = zt + ut_moment + torch.randn_like(xt_moment) * sigma_t_i[0]
        scal = get_scalings_for_boundary_condition(sigma_cm_i)
        c_skip, c_out, c_in = [append_dims(s, xt.ndim) for s in scal]
        et = model(c_in * z_noisy, 1000 * 0.25 * torch.log(sigma_cm_i + 1e-44))
        xt = torch.clamp(c_out * et + c_skip * z_noisy, -1.0, 1.0)
        ut = ut_moment + (zt - xt)
        xt_moment = xt + momentum[iter_ind] * (xt - xt_1)
        ut_moment = ut + momentum[iter_ind] * (ut - ut_1)

    torch.save({"x_raw": xt.cpu(), "picks": picks}, f"{exp}/raw_{state['img']}.pt")
    print(f"[{variant}] img {state['img']} picks {[(p[0], p[1]) for p in picks]}", flush=True)
    return xt


W.pnp_cm_natural_nonlinear_restoration = restoration
M.main()
