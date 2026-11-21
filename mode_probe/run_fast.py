"""PnP-CM nonlinear sampler with the inner data-fit loop swapped for faster solvers.

Subproblem at every outer step (unchanged):  F(z) = 0.5||A(z)-y||^2 + (rho/2)||z - v||^2
Solvers (env SOLVER):
  sgd : the repo's loop, verbatim (fixed count, hand-set lr)              [baseline]
  as  : FAST-DIPS-style steepest descent with the analytic Gauss-Newton-model step
        alpha* = ||g||^2 / (||J g||^2 + rho ||g||^2), then Armijo backtracking.   K steps
  gn  : CLAMP-style damped Gauss-Newton: (J^T J + rho I) d = -g by matrix-free CG
        (rho is the damping), then Armijo backtracking.                  K steps x CG iters
Only the data-step idea of each paper is used; the outer PnP-CM loop is untouched.

Run from the PnP-CM repo root:
  SOLVER=as K=20 python ../mode_probe/run_fast.py <variant> --deg=phase_retrieval [main.py args]
"""
import os, sys, time, json
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath("bkse"))
variant = sys.argv[1]
SOLVER = os.environ.get("SOLVER", "sgd")
K = int(os.environ.get("K", "20"))
CG = int(os.environ.get("CG", "10"))

# SHARD="start:end": run only those images (one shard per GPU).  With SHARD set, the RNG is re-seeded
# per image from its global index, so the measurement noise of image i is identical in every variant
# and every shard layout, whatever the sampler drew before.
SHARD = os.environ.get("SHARD", "")
S0, S1E = (int(t) for t in SHARD.split(":")) if SHARD else (0, 0)
SUF = f"_s{S0}" if SHARD else ""
exp = os.path.abspath(f"../mode_probe/exp_{variant}{SUF}")
os.makedirs(exp, exist_ok=True)
DATA = os.environ.get("DATA", "")  # optional alternative datasets root (e.g. ../mode_probe/data100)
for sub, src in [("logs", "exp/logs"), ("datasets", DATA or "exp/datasets")]:
    if not os.path.exists(f"{exp}/{sub}"):
        os.symlink(os.path.abspath(src), f"{exp}/{sub}")
sys.argv = ["main.py"] + sys.argv[2:] + ([f"--subset_start={S0}", f"--subset_end={S1E}"] if SHARD else []) + ["--path_y=celeba_hq", "--sigma_y=0.05",
                                         "--config=celeba_hq_256.yml", "--save_y", f"--exp={exp}"]

import functions.diff_jpeg as dj


class DiffJPEGAdapter:  # see run_nl.py: our reconstruction of the unreleased wrapper
    def __init__(self, jpeg_qf, device):
        self.q = torch.tensor([float(jpeg_qf)], device=device)

    def A(self, x, **kwargs):
        return dj.diff_jpeg_coding(image_rgb=(x + 1) / 2 * 255, jpeg_quality=self.q.expand(x.shape[0])) / 255 * 2 - 1


dj.DiffJPEGCoding = DiffJPEGAdapter

import main as M
import wrappers.natural_nonlinear_wrapper as W

_dt = W.data_transform
def _stash(config, X):
    if SHARD:
        torch.manual_seed(7000000 + state["img"] + 1)  # called once per image, just before y is drawn
    out = _dt(config, X); state["gt"] = out; return out
W.data_transform = _stash
from pnp_cm_scheme.natural_nonlinear_scheme import (
    get_scalings_for_boundary_condition, append_dims, get_next_alpha, compute_alpha,
    phase_retrieval_initialization)

state = {"img": S0 - 1, "stats": []}


def restoration(y, model, A_funcs, betas, iN, gamma, deltas, rhos, mu_0, classes, args, config,
                sigma_min=0.002, sigma_max=80.0):
    state["img"] += 1
    dev = y.device
    cnt = {"fwd": 0, "vjp": 0, "jvp": 0}
    t_inner = 0.0
    A = A_funcs.A
    penalty_param = nn.Softplus()(torch.linspace(rhos[0], rhos[1], args.T_sampling)).to(dev)
    momentum = (torch.tensor([mu_0]).repeat(args.T_sampling) / torch.arange(1, args.T_sampling + 1)).to(dev)
    shp = (config.sampling.batch_size, config.data.channels, config.data.image_size, config.data.image_size)
    xt = torch.zeros(shp, device=dev); xt_moment = torch.zeros(shp, device=dev)
    ut = torch.zeros(shp, device=dev); ut_moment = torch.zeros(shp, device=dev)
    zt = phase_retrieval_initialization(y, args.phase_ret_os) if args.deg == "phase_retrieval" else y.clone()
    INIT = os.environ.get("INIT", "")
    ginit = torch.Generator(device=dev).manual_seed(4321 + state["img"])
    if INIT == "oracle":   # start near the truth: tests whether the cold start is what breaks GN
        zt = state["gt"].to(dev) + 0.3 * torch.randn(zt.shape, device=dev, generator=ginit)
    elif INIT.startswith("prev:"):  # two-pass: start from the raw output of an earlier (cheap) run
        zt = torch.load(f"../mode_probe/exp_{INIT[5:]}{SUF}/raw_{state['img']}.pt")["x_raw"].to(dev)
        if os.environ.get("SIGNFIX", "0") == "1":  # reference-free: the consistency model picks among the 8 sign flips
            import itertools
            Fl = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3)), device=dev).view(8, 3, 1, 1)
            c = Fl * zt; sg = torch.tensor(0.3, device=dev)
            cs, co, ci = get_scalings_for_boundary_condition(sg); tt = (1000 * 0.25 * torch.log(sg)).repeat(8)
            acc = 0
            for _ in range(4):
                xn = c + sg * torch.randn(c[:1].shape, device=dev, generator=ginit)
                acc = acc + (((co * model(ci * xn, tt) + cs * xn).clamp(-1, 1) - c) ** 2).mean(dim=(1, 2, 3))
            zt = c[int(acc.argmin())][None]
    elif INIT == "randn":  # cold start for tasks that normally start from the measurement
        zt = 0.5 * torch.randn(zt.shape, device=dev, generator=ginit)

    t = (torch.ones(1) * (iN + 1)).to(dev)
    alphas = [compute_alpha(betas, t.long())]
    for _ in range(args.T_sampling - 1):
        alphas.append(get_next_alpha(alphas[-1], gamma).reshape(1))

    def Fval(z, v, rho):
        cnt["fwd"] += 1
        r = A(z) - y
        return 0.5 * (r * r).sum() + 0.5 * rho * ((z - v) ** 2).sum()

    def Fgrad(z, v, rho):
        with torch.enable_grad():
            zr = z.detach().requires_grad_(True)
            f = Fval(zr, v, rho)
            g, = torch.autograd.grad(f, zr)
        cnt["vjp"] += 1
        return f.detach(), g

    def jvp(z, p):
        cnt["jvp"] += 1
        try:
            return torch.func.jvp(A, (z,), (p,))[1]
        except Exception:  # forward-mode unsupported for some op: finite-difference probe
            eps = 1e-3 / (p.norm() + 1e-12)
            cnt["fwd"] += 2
            return (A(z + eps * p) - A(z - eps * p)) / (2 * eps)

    def vjp(z, w):
        with torch.enable_grad():
            zr = z.detach().requires_grad_(True)
            out, = torch.autograd.grad((A(zr) * w).sum(), zr)
        cnt["fwd"] += 1; cnt["vjp"] += 1
        return out

    def armijo(z, d, f0, slope, v, rho, a0):
        a = a0
        for _ in range(12):
            zn = z + a * d
            fn = Fval(zn, v, rho)
            if fn <= f0 + 1e-4 * a * slope:
                return zn
            a *= 0.5
        return z

    def solve_as(z, v, rho):
        for _ in range(K):
            f0, g = Fgrad(z, v, rho)
            gg = (g * g).sum()
            Jg = jvp(z, g)
            a = gg / ((Jg * Jg).sum() + rho * gg + 1e-30)
            z = armijo(z, -g, f0, -gg, v, rho, a)
        return z

    def solve_gn(z, v, rho):
        for _ in range(K):
            f0, g = Fgrad(z, v, rho)
            d = torch.zeros_like(z); r = -g.clone(); p = r.clone(); rs = (r * r).sum()
            for _ in range(CG):
                Hp = vjp(z, jvp(z, p)) + rho * p
                a = rs / ((p * Hp).sum() + 1e-30)
                d = d + a * p; r = r - a * Hp
                rs_new = (r * r).sum()
                if rs_new < 1e-10 * (g * g).sum():
                    break
                p = r + (rs_new / rs) * p; rs = rs_new
            z = armijo(z, d, f0, (g * d).sum(), v, rho, 1.0)
        return z

    def solve_sgd(z, v, rho):
        loss_pre = torch.tensor(0.0, device=dev)
        with torch.enable_grad():
            ztmp = z.clone().requires_grad_(True)
            optimizer = getattr(optim, args.opt_type)([ztmp], lr=float(args.opt_lr))
            for _ in range(args.opt_num_iter):
                res = A(ztmp) - y
                diff = ztmp - v
                loss = 0.5 * (res * res).sum() + 0.5 * rho * torch.sum(diff * diff).abs()
                loss.backward()
                optimizer.step()
                cnt["fwd"] += 1; cnt["vjp"] += 1
                if loss.item() > loss_pre.item():
                    for g in optimizer.param_groups:
                        g["lr"] = g["lr"] * args.opt_decay_rate
                loss_pre = loss
        return ztmp.detach()

    # ---- combined solver -------------------------------------------------------------
    # tau: data-fit value explained by measurement noise alone, 0.5 * m * sigma_y^2.
    # step 1: first-order steps (repo optimizer) until misfit <= tau (cap: repo count);
    # every step: damped Gauss-Newton, stopped when the objective no longer drops by TOL.
    tau = 0.5 * y.numel() * float(args.sigma_y) ** 2
    COLD = float(os.environ.get("COLD", "10")); TOL = float(os.environ.get("TOL", "1e-2"))
    trace = []

    def solve_auto(z, v, rho, first=False):
        cnt["fwd"] += 1
        m0 = float(0.5 * ((A(z) - y) ** 2).sum())
        n_fo = n_gn = 0
        # first outer step has no anchor (v = 0, rho tiny): a pure non-convex data fit, so use
        # first-order steps there, but only until the misfit reaches the noise level tau
        if first:
            loss_pre = torch.tensor(0.0, device=dev)
            with torch.enable_grad():
                ztmp = z.clone().requires_grad_(True)
                optimizer = getattr(optim, args.opt_type)([ztmp], lr=float(args.opt_lr))
                for _ in range(args.opt_num_iter):
                    res = A(ztmp) - y
                    mis = 0.5 * (res * res).sum()
                    if mis.item() <= tau:
                        break
                    loss = mis + 0.5 * rho * torch.sum((ztmp - v) ** 2)
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    cnt["fwd"] += 1; cnt["vjp"] += 1; n_fo += 1
                    if loss.item() > loss_pre.item():
                        for g in optimizer.param_groups:
                            g["lr"] = g["lr"] * args.opt_decay_rate
                    loss_pre = loss
            z = ztmp.detach()
        f_prev = None
        for _ in range(K):
            f0, g = Fgrad(z, v, rho)
            if f_prev is not None and (f_prev - f0) < TOL * f_prev:
                break
            f_prev = f0
            d = torch.zeros_like(z); r = -g.clone(); p = r.clone(); rs = (r * r).sum()
            for _ in range(CG):
                Hp = vjp(z, jvp(z, p)) + rho * p
                a = rs / ((p * Hp).sum() + 1e-30)
                d = d + a * p; r = r - a * Hp
                rs_new = (r * r).sum()
                if rs_new < 1e-4 * (g * g).sum():
                    break
                p = r + (rs_new / rs) * p; rs = rs_new
            z = armijo(z, d, f0, (g * d).sum(), v, rho, 1.0)
            n_gn += 1
        trace.append({"ratio0": m0 / tau, "first_order": n_fo, "gn": n_gn})
        return z

    # ---- step-1 experiments ---------------------------------------------------------
    S1 = os.environ.get("S1", ""); S1_N = int(os.environ.get("S1_N", "1000")); S1_LR = float(os.environ.get("S1_LR", "0.4"))
    s1log = {}

    def quality(z):  # sign-invariant agreement with ground truth on pixels the mask keeps (diagnostic only)
        if args.deg != "phase_retrieval":
            return float("nan")
        gt = state["gt"].to(dev); keep = (A_funcs.modulation != 0)
        q = []
        for c in range(3):
            a_, b_ = z[0, c][keep], gt[0, c][keep]
            a_ = a_ - a_.mean(); b_ = b_ - b_.mean()
            q.append(abs(float((a_ * b_).sum() / (a_.norm() * b_.norm() + 1e-12))))
        return sum(q) / 3

    def log_point(tr, k, z, z_prev, f, g):
        tr.append({"k": k, "misfit": float(0.5 * ((A(z) - y) ** 2).sum()), "obj": float(f), "gnorm": float(g.norm()),
                   "dz": float((z - z_prev).norm()), "q": quality(z)})

    def step1(z, v, rho):
        tr = []; ups = 0; z_prev = z.clone()
        if S1 in ("sgd", "sgdmono"):      # plain gradient steps with a fixed lr (repo rule: lr*=0.95 when loss rises)
            lr = S1_LR; f_pre = None
            for k in range(S1_N):
                f, g = Fgrad(z, v, rho)
                if k % 10 == 0: log_point(tr, k, z, z_prev, f, g)
                if f_pre is not None and f > f_pre:
                    ups += 1; lr *= 0.95
                    if S1 == "sgdmono":   # monotone variant: undo the step that raised the loss
                        z = z_prev.clone(); f, g = Fgrad(z, v, rho)
                f_pre = f; z_prev = z.clone(); z = z - lr * g
            s1log["lr_end"] = lr
        elif S1 == "accum":               # the repo loop exactly: backward() without zero_grad, so gradients ACCUMULATE
            lr = S1_LR; f_pre = 0.0; acc = torch.zeros_like(z)
            for k in range(S1_N):
                f, g = Fgrad(z, v, rho)
                if k % 10 == 0: log_point(tr, k, z, z_prev, f, g)
                acc = acc + g; z_prev = z.clone(); z = z - lr * acc
                if float(f) > f_pre:
                    ups += 1; lr *= 0.95
                f_pre = float(f)
            s1log["lr_end"] = lr
        elif S1 == "hb":                  # heavy ball with fixed momentum S1_B and fixed lr
            beta = float(os.environ.get("S1_B", "0.9")); vel = torch.zeros_like(z); f_pre = None
            for k in range(S1_N):
                f, g = Fgrad(z, v, rho)
                if k % 10 == 0: log_point(tr, k, z, z_prev, f, g)
                if f_pre is not None and f > f_pre: ups += 1
                f_pre = f; vel = beta * vel + g; z_prev = z.clone(); z = z - S1_LR * vel
        elif S1 == "nag":                 # Nesterov + function-value restart; step = analytic step at the start (no lr)
            f, g = Fgrad(z, v, rho); gg = (g * g).sum(); Jg = jvp(z, g)
            a = float(gg / ((Jg * Jg).sum() + rho * gg + 1e-30)); s1log["step"] = a
            x_prev = z.clone(); yk = z.clone(); t = 0; f_pre = float(f)
            for k in range(S1_N):
                f, g = Fgrad(yk, v, rho)
                if k % 10 == 0: log_point(tr, k, yk, z_prev, f, g)
                z_prev = yk.clone()
                x_new = yk - a * g
                if float(f) > f_pre:      # restart momentum when the objective rises
                    ups += 1; t = 0
                f_pre = float(f); t += 1
                yk = x_new + (t - 1) / (t + 2) * (x_new - x_prev); x_prev = x_new
            z = x_prev
        elif S1 == "bb":                  # Barzilai-Borwein: self-chosen steps, no decrease check, no lr
            f, g = Fgrad(z, v, rho); gg = (g * g).sum(); Jg = jvp(z, g)
            a = float(gg / ((Jg * Jg).sum() + rho * gg + 1e-30)); f_pre = f
            for k in range(S1_N):
                if k % 10 == 0: log_point(tr, k, z, z_prev, f, g)
                z_prev = z.clone(); g_prev = g
                z = z - a * g
                f, g = Fgrad(z, v, rho)
                if f > f_pre: ups += 1
                f_pre = f
                sk = z - z_prev; yk = g - g_prev
                a = float((sk * sk).sum() / ((sk * yk).sum().abs() + 1e-30))
        elif S1 == "asfree":              # analytic step every iteration, accepted without a decrease check
            f_pre = None
            for k in range(S1_N):
                f, g = Fgrad(z, v, rho)
                if k % 10 == 0: log_point(tr, k, z, z_prev, f, g)
                if f_pre is not None and f > f_pre: ups += 1
                f_pre = f; gg = (g * g).sum(); Jg = jvp(z, g)
                z_prev = z.clone(); z = z - (gg / ((Jg * Jg).sum() + rho * gg + 1e-30)) * g
        s1log["ups"] = ups; s1log["trace"] = tr
        return z

    solvers = {"sgd": solve_sgd, "as": solve_as, "gn": solve_gn, "auto": solve_auto}
    HYB = int(os.environ.get("HYBRID", "0"))  # first HYB outer steps use the repo SGD loop
    fin = []
    z_init = zt.clone()  # the image this pass starts from (measurement-based start, or a previous pass)
    START = int(os.environ.get("START", "0"))  # skip the first START outer steps (short second pass)
    if START > 0:
        xt = zt.clone(); xt_moment = zt.clone()  # anchor the first executed step on the warm image
    for iter_ind, at in enumerate(alphas):
        if iter_ind < START:
            continue
        sigma_t_i = torch.sqrt(1 - at)
        sigma_cm_i = torch.clamp((1 + deltas[iter_ind]) * sigma_t_i, sigma_min, sigma_max)
        sigma_t_i = torch.clamp(sigma_t_i, sigma_min, sigma_max)
        xt_1, ut_1 = xt.clone(), ut.clone()
        v = xt_moment - ut_moment
        torch.cuda.synchronize(); t0 = time.time()
        if S1 and iter_ind == 0:
            zt = step1(zt, v, penalty_param[iter_ind])
        elif SOLVER == "auto":
            zt = solve_auto(zt, v, penalty_param[iter_ind], first=(iter_ind == 0))
        else:
            zt = solvers["sgd" if iter_ind < HYB else SOLVER](zt, v, penalty_param[iter_ind])
        torch.cuda.synchronize(); t_inner += time.time() - t0
        fin.append(float(0.5 * ((A(zt) - y) ** 2).sum()))

        z_noisy = zt + ut_moment + torch.randn_like(xt_moment) * sigma_t_i[0]
        c_skip, c_out, c_in = [append_dims(s, xt.ndim) for s in get_scalings_for_boundary_condition(sigma_cm_i)]
        et = model(c_in * z_noisy, 1000 * 0.25 * torch.log(sigma_cm_i + 1e-44))
        xt = torch.clamp(c_out * et + c_skip * z_noisy, -1.0, 1.0)
        ut = ut_moment + (zt - xt)
        xt_moment = xt + momentum[iter_ind] * (xt - xt_1)
        ut_moment = ut + momentum[iter_ind] * (ut - ut_1)

    # ground-truth-free diagnostics of this pass (candidates for an automatic "second pass or not" rule)
    gd = torch.Generator(device=dev).manual_seed(55 + state["img"])
    sg = torch.tensor(0.3, device=dev); cs_, co_, ci_ = get_scalings_for_boundary_condition(sg)
    pr = 0.0
    for _ in range(2):
        xn = xt + sg * torch.randn(xt.shape, device=dev, generator=gd)
        pr += float((((co_ * model(ci_ * xn, (1000 * 0.25 * torch.log(sg)).reshape(1)) + cs_ * xn).clamp(-1, 1) - xt) ** 2).mean()) / 2
    diag = {"mis_init": float(0.5 * ((A(z_init) - y) ** 2).sum()), "mis_out": float(0.5 * ((A(xt) - y) ** 2).sum()),
            "prior_out": pr, "move": float((xt - z_init).norm() / (xt.norm() + 1e-12)),
            "move_abs": float(((xt - z_init) ** 2).mean().sqrt())}
    rec = {"img": state["img"], "t_inner": t_inner, **cnt, "datafit_per_step": fin, "tau": tau, "trace": trace, "s1": dict(s1log), "diag": diag}
    state["stats"].append(rec)
    torch.save({"x_raw": xt.cpu(), "picks": []}, f"{exp}/raw_{state['img']}.pt")
    json.dump(state["stats"], open(f"{exp}/stats.json", "w"))
    print(f"[{variant}] img {state['img']} inner {t_inner:.1f}s fwd {cnt['fwd']} vjp {cnt['vjp']} jvp {cnt['jvp']} "
          f"datafit {fin[0]:.1f} -> {fin[-1]:.1f}", flush=True)
    return xt


W.pnp_cm_natural_nonlinear_restoration = restoration
M.main()
