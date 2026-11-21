"""A theory-derived alternative to PnP-CM's step: split Gibbs sampling with one number per step.

Model: p(x, z | y) ~ p(x) * exp(-||A z - y||^2 / (2 sigma_y^2)) * exp(-||x - z||^2 / (2 eta^2)).
Alternating the two exact conditionals gives, for coupling level eta_k at step k:
  likelihood step   z ~ N(m, C),  C = (A^T A / sigma_y^2 + I / eta^2)^-1,  m = C (A^T y / sigma_y^2 + x / eta^2)
                    -> the penalty is DERIVED: rho = sigma_y^2 / eta^2, and the injected noise is the
                       operator-shaped Gaussian with covariance C (std eta where A sees nothing, ~sigma_y/s where
                       it sees well); no separate injected sigma, no delta, no dual variable, no momentum
  prior step        x ~ p(x | z = x + eta * noise)  -> approximated by one CM call, x = CM(z, c * eta)
So a T-step method has T numbers (eta_1..eta_T), or 2 with a geometric schedule, plus optionally c.

Variants (linear tasks; everything chosen on the 8 calibration images, scored on 10 test images x 3 seeds):
  shipped-levels   eta_k = the shipped levels told to the CM, c = 1: zero numbers searched
  geometric T=2,3,4   eta from eta_first to eta_last geometrically; grid over (eta_first, eta_last, c)
  no-noise         best geometric setting but z = m (argmin instead of a sample)

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/gibbs.py <task>
"""
import os, sys, json, itertools
import numpy as np, torch

TASK = sys.argv[1]
sys.argv = ["steps.py", "lib", TASK]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steps as S

R, A, dev, SHP, SY = S.R, S.A, S.dev, S.SHP, S.SIG_Y
sing = A.add_zeros(A.singulars()[None])[0]
q = lambda p, l: p - l / 0.03


@torch.no_grad()
def gibbs(y, xinit, etas, seed, c=1.0, sample=True):
    torch.manual_seed(seed)
    x, ATy = xinit.clone(), A.At(y).view(SHP)
    for eta in etas:
        rho = SY ** 2 / eta ** 2
        z = A.AtA_add_eta_inv(ATy + rho * x, eta=rho).reshape(SHP)
        if sample:
            std = eta * SY / torch.sqrt(sing ** 2 * eta ** 2 + SY ** 2)
            z = z + A.V((std * torch.randn(sing.numel(), device=dev))[None]).reshape(SHP)
        x = S.cm(z, float(np.clip(c * eta, 0.002, 80)))
    return x


def ev(data, inits, etas, c, sample=True, seeds=(0,)):
    sc = [S.score(gibbs(y, inits[i], etas, i + 1000 * sd, c, sample), x0) for sd in seeds for i, (x0, y) in enumerate(data)]
    return float(np.mean([s[0] for s in sc])), float(np.mean([s[1] for s in sc]))


geo = lambda a, b, T: [float(v) for v in np.exp(np.linspace(np.log(a), np.log(b), T))]
print(f"\n######## {TASK}: split-Gibbs step with a derived penalty and operator-shaped noise")
told = [s["scm"] for s in S.SHIP]
print("consistency check on the SHIPPED numbers (theory says rho = sigma_y^2 / level^2):")
print("  level told to CM      " + "  ".join(f"{t:.3f}" for t in told))
print("  level implied by rho  " + "  ".join(f"{SY / s['rho'] ** 0.5:.3f}" for s in S.SHIP))
ref = S.test_eval(S.SHIP, "white", None)
rows = {"shipped PnP-CM (11 numbers)": ref}
print(f"\n{'method':34s} numbers searched | test PSNR / LPIPS")
print(f"{'shipped PnP-CM, 4 steps':34s} {'(11 hand-set)':>16s} | {ref[0]:.2f} / {ref[1]:.3f}")
t = ev(R.data, S.TEST_INIT, told, 1.0, seeds=(0, 1, 2))
rows["gibbs shipped-levels"] = t
print(f"{'Gibbs, shipped levels, 4 steps':34s} {'0':>16s} | {t[0]:.2f} / {t[1]:.3f}", flush=True)
for T in (2, 3, 4):
    best = None
    for a, b, c in itertools.product((0.2, 0.3, 0.45, 0.6, 0.8, 1.0), (0.01, 0.02, 0.04, 0.08), (1.0, 1.2)):
        p, l = ev(S.cal, S.CAL_INIT, geo(a, b, T), c, seeds=(100,))
        if best is None or q(p, l) > best[0]:
            best = (q(p, l), a, b, c)
    _, a, b, c = best
    t = ev(R.data, S.TEST_INIT, geo(a, b, T), c, seeds=(0, 1, 2))
    rows[f"gibbs geometric T={T}"] = dict(test=t, eta_first=a, eta_last=b, c=c)
    print(f"{f'Gibbs, geometric, {T} steps':34s} {'3':>16s} | {t[0]:.2f} / {t[1]:.3f}   eta {a} -> {b}, c={c}", flush=True)
    if T == 3:
        n = ev(R.data, S.TEST_INIT, geo(a, b, T), c, sample=False, seeds=(0, 1, 2))
        rows["gibbs geometric T=3 no-noise"] = n
        print(f"{'  same, argmin instead of sample':34s} {'':>16s} | {n[0]:.2f} / {n[1]:.3f}", flush=True)
json.dump(rows, open(os.path.abspath(f"../mode_probe/random_steps/steps/{TASK}_gibbs.json"), "w"))
