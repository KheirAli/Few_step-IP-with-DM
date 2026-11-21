"""Can the noise level given to the CM be ESTIMATED instead of tuned?

Runs the shipped PnP-CM schedule and, at every step, intercepts the CM input z_noisy and compares:
  injected   sigma_t actually added
  shipped    sigma_cm = (1+delta)*sigma_t given to the CM (hand-tuned)
  true_rms   rms(z_noisy - ground truth): the real total corruption of the CM input
  struct     rms(z + u - ground truth): the non-Gaussian part (before noise is added)
  oracle     the sigma that minimizes ||CM(z_noisy, sigma) - ground truth|| (best level in hindsight)
  fixedpt    ground-truth-free estimate: the sigma where rms(z_noisy - CM(z_noisy, sigma)) = sigma

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/noise_probe.py <task> [n_images]
"""
import os, sys, json
import numpy as np, torch

task, n_img = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
sys.argv = ["random_steps.py", task, "--lib"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_steps as R
from pnp_cm_scheme.natural_linear_scheme import get_scalings_for_boundary_condition

net, dev = R.model, R.dev
GRID = np.exp(np.linspace(np.log(0.004), np.log(3.0), 28))
rms = lambda v: float(v.pow(2).mean().sqrt())
seen = []


def recorder(xin, t):
    scm = float(torch.exp(t.flatten()[0] / 250.0))
    seen.append((xin.detach() * (scm ** 2 + 0.25) ** 0.5, scm))  # undo c_in -> z_noisy
    return net(xin, t)


@torch.no_grad()
def cm(z, sigma):
    s = torch.tensor(sigma, device=dev, dtype=torch.float32)
    c_skip, c_out, c_in = get_scalings_for_boundary_condition(s)
    return (c_out * net(c_in * z, (250 * torch.log(s)).reshape(1)) + c_skip * z).clamp(-1, 1)


def crossing(r):
    """largest sigma on the grid where the residual curve r(sigma) crosses the line r = sigma."""
    d = np.log(r) - np.log(GRID)
    for k in range(len(GRID) - 1, 0, -1):
        if d[k - 1] > 0 >= d[k]:
            w = d[k - 1] / (d[k - 1] - d[k])
            return float(np.exp(np.log(GRID[k - 1]) * (1 - w) + np.log(GRID[k]) * w))
    return float("nan")


R.model = recorder
ship = R.configs_all[0]
rows = []
for i, (x0, y) in enumerate(R.data[:n_img]):
    seen.clear()
    torch.manual_seed(i)
    with torch.no_grad():
        R.restore(ship, y)
    for k, (z, scm) in enumerate(seen):
        st = ship["sigma_t"][k]
        err = np.array([rms(cm(z, s) - x0) for s in GRID])
        res = np.array([rms(z - cm(z, s)) for s in GRID])
        tot = rms(z - x0)
        rows.append(dict(img=i, step=k, injected=st, shipped=scm, true_rms=tot,
                         struct=float(max(tot ** 2 - st ** 2, 0) ** 0.5), oracle=float(GRID[err.argmin()]),
                         fixedpt=crossing(res), err_at_shipped=rms(cm(z, scm) - x0), err_at_oracle=float(err.min()),
                         err_at_true=rms(cm(z, tot) - x0)))

out = os.path.abspath(f"../mode_probe/random_steps/noise_probe_{task}.json")
json.dump(rows, open(out, "w"))
print(f"\n=== {task}: shipped schedule, mean over {n_img} test images (sigma units: images in [-1,1])")
print("step | injected  shipped | true_rms  struct | oracle  fixedpt | CM-output error at: shipped  true_rms  oracle")
for k in range(R.T):
    g = [r for r in rows if r["step"] == k]
    a = lambda key: np.nanmean([r[key] for r in g])
    print(f"{k + 1:4d} | {a('injected'):8.3f} {a('shipped'):8.3f} | {a('true_rms'):8.3f} {a('struct'):7.3f} | "
          f"{a('oracle'):6.3f} {a('fixedpt'):8.3f} | {a('err_at_shipped'):27.4f} {a('err_at_true'):9.4f} {a('err_at_oracle'):7.4f}")
