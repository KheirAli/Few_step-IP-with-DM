"""Structural limits of PnP-CM, measured.

A  The CM as a denoiser in the ideal case: input = ground truth + white Gaussian noise at the stated level.
   This is the best any step can do at that noise level; nothing in the loop can beat it.
B  PnP-CM returns one random sample.  Average K runs with different sampler seeds: PSNR rises toward the
   posterior-mean value and sharpness drops.  Also: does the per-pixel spread across seeds predict the error?
C  Stability: local gain of the CM, ||CM(a) - CM(b)|| / ||a - b|| for a small random change of the input.
   The paper's convergence theorem assumes a Lipschitz denoiser; a gain above 1 means the CM amplifies changes.

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/ceiling.py <task>
"""
import os, sys, json
import numpy as np, torch

TASK = sys.argv[1]
sys.argv = ["steps.py", "lib", TASK]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steps as S

R, dev = S.R, S.dev
rms = lambda v: float(v.pow(2).mean().sqrt())
out = {}
print(f"\n######## {TASK}")
if TASK == "deblur_gauss":
    print("A/C  CM alone on ground truth + white noise (10 test images)")
    print("  sigma   PSNR   LPIPS | local gain for a small input change")
    with torch.no_grad():
        for sg in (0.01, 0.03, 0.08, 0.16, 0.33, 0.43, 0.7, 1.0):
            ps, lp, gain = [], [], []
            for i, (x0, _) in enumerate(R.data):
                torch.manual_seed(i)
                z = x0 + sg * torch.randn_like(x0)
                a = S.cm(z, sg)
                d = 0.01 * torch.randn_like(x0)
                gain.append(rms(S.cm(z + d, sg) - a) / rms(d))
                ps.append(R.psnr(a, x0)); lp.append(float(R.lp(a, x0)))
            out[f"denoise_{sg}"] = (np.mean(ps), np.mean(lp), np.mean(gain))
            print(f"  {sg:5.2f}  {np.mean(ps):5.2f}  {np.mean(lp):.3f} | {np.mean(gain):.2f}")

print("B  averaging K shipped runs with different sampler seeds (10 test images)")
K = 8
runs = []
for sd in range(K):
    xs = []
    for i, (x0, y) in enumerate(R.data):
        x, _ = S.pnp(y, S.TEST_INIT[i], S.SHIP, "white", i + 1000 * sd)
        xs.append(x)
    runs.append(torch.cat(xs))
runs = torch.stack(runs)  # K, 10, 3, H, W
gt = torch.cat([x0 for x0, _ in R.data])
with torch.no_grad():
    for k in (1, 2, 4, 8):
        m = runs[:k].mean(0)
        ps = np.mean([R.psnr(m[i:i + 1], gt[i:i + 1]) for i in range(len(gt))])
        lp = np.mean([float(R.lp(m[i:i + 1], gt[i:i + 1])) for i in range(len(gt))])
        out[f"avg_{k}"] = (ps, lp)
        print(f"  K={k}: PSNR {ps:.2f}  LPIPS {lp:.3f}")
sd_map, err = runs.std(0), (runs.mean(0) - gt).abs()
c = np.mean([float(np.corrcoef(sd_map[i].flatten().cpu(), err[i].flatten().cpu())[0, 1]) for i in range(len(gt))])
print(f"  spread across seeds: rms {rms(sd_map):.3f}; error of a single run: rms {rms(runs[0] - gt):.3f}; "
      f"error of the 8-run mean: rms {rms(runs.mean(0) - gt):.3f}")
print(f"  per-pixel correlation between seed spread and actual error of the mean: {c:.2f}")
out["spread"] = (rms(sd_map), rms(runs[0] - gt), rms(runs.mean(0) - gt), c)
json.dump(out, open(os.path.abspath(f"../mode_probe/random_steps/steps/{TASK}_ceiling.json"), "w"))
