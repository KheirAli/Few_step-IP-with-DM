"""Does each image need its own noise level?  2-step PnP-CM with shaped noise, linear tasks.

Base = the first two shipped steps.  The noise of step 1 is scaled by a and of step 2 by b (injected sigma
and the level told to the CM together), a, b in {0.5, 0.7, 1, 1.4, 2}: 25 schedules.
  global     one (a, b) for the whole task, chosen on the 8 calibration images
  per-image  the best (a, b) for each test image, chosen WITH its ground truth on sampler seeds 0 and 1
             and scored on the unseen seed 2 (an upper bound on what any per-image rule could reach)
Quality score used for choosing: PSNR - LPIPS / 0.03  (1 dB counted like 0.03 LPIPS).

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/per_image.py <task>
"""
import os, sys, json
import numpy as np, torch

TASK = sys.argv[1]
sys.argv = ["steps.py", "lib", TASK]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steps as S

R, GRID = S.R, [0.5, 0.7, 1.0, 1.4, 2.0]
q = lambda p, l: p - l / 0.03
combos = [(a, b) for a in GRID for b in GRID]
cal_q = np.zeros(len(combos))
test = np.zeros((len(combos), 3, len(R.data), 2))  # combo, seed, image, (psnr, lpips)
for c, (a, b) in enumerate(combos):
    st = [dict(S.SHIP[0]), dict(S.SHIP[1])]
    for s, f in zip(st, (a, b)):
        s["st"], s["scm"] = s["st"] * f, s["scm"] * f
    cp, cl, tab = S.cal_eval(st, "shaped", True)
    cal_q[c] = q(cp, cl)
    for sd in range(3):
        for i, (x0, y) in enumerate(R.data):
            x, _ = S.pnp(y, S.TEST_INIT[i], st, "shaped", i + 1000 * sd, tables=tab)
            test[c, sd, i] = S.score(x, x0)

tq = q(test[..., 0], test[..., 1])  # combo, seed, image
g = int(np.argmax(cal_q))
pick = np.argmax(tq[:, :2].mean(1), axis=0)  # per image, from seeds 0 and 1
n = len(R.data)
per = np.array([test[pick[i], 2, i] for i in range(n)])
glob = test[g, 2]
base = test[combos.index((1.0, 1.0)), 2]
best_single = int(np.argmax(tq[:, :2].mean((1, 2))))  # best single schedule in hindsight on test
print(f"\n######## {TASK}: 2 steps, shaped noise, scored on sampler seed 2 over the {n} test images")
print(f"  shipped first two steps (a=b=1)            PSNR {base[:, 0].mean():.2f}  LPIPS {base[:, 1].mean():.3f}")
print(f"  global, chosen on calibration  a={combos[g][0]} b={combos[g][1]}   PSNR {glob[:, 0].mean():.2f}  LPIPS {glob[:, 1].mean():.3f}")
print(f"  best single schedule in hindsight a={combos[best_single][0]} b={combos[best_single][1]} PSNR {test[best_single, 2, :, 0].mean():.2f}  "
      f"LPIPS {test[best_single, 2, :, 1].mean():.3f}")
print(f"  per-image oracle                           PSNR {per[:, 0].mean():.2f}  LPIPS {per[:, 1].mean():.3f}")
print("  per-image best (a, b): " + "  ".join(f"{combos[p][0]}/{combos[p][1]}" for p in pick))
print(f"  images whose best differs from the global choice: {int(np.sum(pick != g))} of {n}")
d = per - glob
print(f"  per-image gain over global: PSNR {d[:, 0].mean():+.2f} dB (largest single image {d[:, 0].max():+.2f}), LPIPS {d[:, 1].mean():+.3f}")
json.dump(dict(combos=combos, cal_q=cal_q.tolist(), test=test.tolist()),
          open(os.path.abspath(f"../mode_probe/random_steps/steps/{TASK}_per_image.json"), "w"))
