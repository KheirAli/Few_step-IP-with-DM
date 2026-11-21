"""Is the plain L2 data-fit step right, especially in the last iterations?  (linear tasks)

The data-fit step solves  min_z ||A z - y||^2 + rho ||z - (x - u)||^2  with a hand-set rho per step.
With Gaussian measurement noise the residual of the TRUE image is ||A x* - y||^2 = m * sigma_y^2.
A residual far below that means the step is fitting the measurement noise.

Part 1: residual ratio ||A z - y||^2 / (m sigma_y^2) of the shipped schedule at each step (1 = noise level).
Part 2: replace the hand-set rho by the discrepancy principle (rho chosen per image and step so the ratio is
        exactly 1; no tuning), in the last 1, last 2, or all steps.  Shaped noise, shipped noise levels.

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/l2_probe.py <task>
"""
import os, sys, json
import numpy as np, torch

TASK = sys.argv[1]
sys.argv = ["steps.py", "lib", TASK]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steps as S

T = len(S.SHIP)
print(f"\n######## {TASK}")
S.DIAG = []
ref = S.test_eval(S.SHIP, "white", None, seeds=1)
d = np.array(S.DIAG); S.DIAG = None
print("Part 1  shipped schedule: residual / expected-noise ratio after the data-fit step (1.0 = fits y to the noise level)")
print("  step:      " + "  ".join(f"{k + 1}: {d[d[:, 0] == k, 1].mean():.2f}" for k in range(T)))
print("  shipped rho " + "  ".join(f"{k + 1}: {S.SHIP[k]['rho']:.3g}" for k in range(T)))

print("Part 2  test PSNR / LPIPS (10 images, 3 seeds), shaped noise")
rows = {}
for name, which in (("shipped rho", []), ("discrepancy, last step", [T - 1]), ("discrepancy, last 2 steps", [T - 2, T - 1]),
                    ("discrepancy, all steps", list(range(T)))):
    st = [dict(s, disc=(k in which)) for k, s in enumerate(S.SHIP)]
    tab = S.cal_eval(st, "shaped", True)[2]
    S.DIAG = []
    S.test_eval(st, "shaped", tab, seeds=1)
    dd = np.array(S.DIAG); S.DIAG = None
    p, l = S.test_eval(st, "shaped", tab)
    rows[name] = (p, l)
    print(f"  {name:26s} {p:.2f} / {l:.3f} | rho used per step (median): "
          + "  ".join(f"{np.median(dd[dd[:, 0] == k, 2]):.3g}" for k in range(T)), flush=True)
json.dump(rows, open(os.path.abspath(f"../mode_probe/random_steps/steps/{TASK}_l2_probe.json"), "w"))
