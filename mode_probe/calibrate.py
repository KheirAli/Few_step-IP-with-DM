"""Calibrate PnP-CM's noise schedule on a few held-out images, then score on the 10 test images.

Calibration images: CelebA-HQ 256 PNGs from CAL_DIR that are not near-duplicates of any test image
(8 picked with a fixed seed; "cal4" is the first 4 of them).  Only the noise schedule is tuned:
sigma_t (injected) and sigma_cm (given to the CM) per step, i.e. 2*T numbers.  rhos, mu_0 and the
data-fit optimizer stay at the shipped values.

Modes
  pool   score the existing random_steps.py schedules on the 8 calibration images (their test
         scores already exist), so calibration->test transfer can be measured.
  es     elitist evolution strategy in log-sigma space.  --obj psnr | lpips | pareto, where pareto
         maximizes min(PSNR gain, LPIPS gain / 0.03) over the shipped schedule on the calibration images:
           --init scratch  start from random schedules (no knowledge of the shipped values)
           --init shipped  start from the shipped schedule
         The winner is then run on the 10 test images with several sampler seeds.

Run from the PnP-CM repo root:
  CUDA_VISIBLE_DEVICES=g python ../mode_probe/calibrate.py <task> --mode es --init scratch --ncal 8
"""
import os, sys, json, glob, time, argparse
import numpy as np, torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("task")
p.add_argument("--mode", default="es")
p.add_argument("--worker", type=int, default=0)
p.add_argument("--nworkers", type=int, default=1)
p.add_argument("--init", default="scratch")
p.add_argument("--ncal", type=int, default=8)
p.add_argument("--obj", default="psnr")
p.add_argument("--gens", type=int, default=40)
p.add_argument("--pop", type=int, default=12)
p.add_argument("--test_seeds", type=int, default=5)
a = p.parse_args()

sys.argv = ["random_steps.py", a.task, "--lib"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_steps as R

CAL_DIR = "/usr/scratch/akheirandish3/diffgeo/data/celebahq256"
dev, T = R.dev, R.T
out = os.path.abspath(f"../mode_probe/random_steps/{a.task}/cal")
os.makedirs(out, exist_ok=True)

# ---- calibration images, disjoint from the test images
files = sorted(glob.glob(CAL_DIR + "/*.png"))
imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).float().permute(2, 0, 1) / 255
                    for f in files]).to(dev)
test = torch.cat([(x0 + 1) / 2 for x0, _ in R.data])
mse = torch.stack([((imgs - t[None]) ** 2).mean(dim=(1, 2, 3)) for t in test], 1).min(1).values
ok = np.where((mse > 1e-3).cpu().numpy())[0]  # drop anything within 30 dB of a test image
pick = np.random.default_rng(0).permutation(ok)[:8]
print(f"calibration pool {len(files)} images, {len(files) - len(ok)} dropped as near-duplicates of test; "
      f"using {[os.path.basename(files[i]) for i in pick]}", flush=True)
cal = R.make_data([imgs[i][None] for i in pick], 7000)


def mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


if a.mode == "pool":
    for cfg in R.configs_all[a.worker::a.nworkers]:
        f = f"{out}/c{cfg['id']:03d}.json"
        if os.path.exists(f):
            continue
        t0 = time.time()
        cfg["cal_rows"] = R.run(cfg, data=cal)
        json.dump(cfg, open(f, "w"))
        print(f"POOL {a.task} c{cfg['id']:03d} {cfg['cond']:13s} cal PSNR {mean(cfg['cal_rows'], 'psnr'):.2f} "
              f"LPIPS {mean(cfg['cal_rows'], 'lpips'):.3f} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit()

# ---------------------------------------------------------------- evolution strategy
tag = f"es_{a.init}_cal{a.ncal}_{a.obj}"
rng = np.random.default_rng({"scratch": 1, "shipped": 2}[a.init] * 100 + a.ncal + {"psnr": 0, "lpips": 50, "pareto": 70}[a.obj])
LO, HI = np.log(0.002), np.log(80.0)
hist = []


def cfg_of(v, seed=0):
    return dict(id=0, cond=tag, mode="sigma", seed=seed, sigma_t=np.exp(v[:T]).tolist(), sigma_cm=np.exp(v[T:]).tolist())


def evaluate(v):
    rows = R.run(cfg_of(v), data=cal[:a.ncal])
    ps, lp = mean(rows, "psnr"), mean(rows, "lpips")
    if a.obj == "pareto":  # worst-case gain over the shipped schedule, 1 dB PSNR == 0.03 LPIPS; >0 means both improved
        s = min(ps - REF[0], (REF[1] - lp) / 0.03)
    else:
        s = ps if a.obj == "psnr" else -lp
    s = s if np.isfinite(s) else -1e9
    hist.append(dict(v=v.tolist(), psnr=ps, lpips=lp, score=s))
    return s


ship = R.configs_all[0]
if a.obj == "pareto":
    _r = R.run(ship, data=cal[:a.ncal])
    REF = (mean(_r, "psnr"), mean(_r, "lpips"))
    print(f"shipped on cal{a.ncal}: PSNR {REF[0]:.2f} LPIPS {REF[1]:.3f}", flush=True)
v_ship = np.log(np.concatenate([ship["sigma_t"], ship["sigma_cm"]]))
if a.init == "shipped":
    elite = [(evaluate(v_ship), v_ship)]
else:
    elite = []
    for _ in range(a.pop):
        st = rng.uniform(np.log(0.01), np.log(1.0), T)
        v = np.concatenate([st, st + np.log1p(rng.uniform(0.0, 0.5, T))])
        elite.append((evaluate(v), v))
elite = sorted(elite, key=lambda e: -e[0])[:4]
step = 0.4
for g in range(a.gens):
    kids = []
    for _ in range(a.pop):
        par = elite[rng.integers(len(elite))][1]
        v = np.clip(par + rng.normal(0, step, 2 * T), LO, HI)
        kids.append((evaluate(v), v))
    elite = sorted(elite + kids, key=lambda e: -e[0])[:4]
    step = max(step * 0.9, 0.05)
    b = cfg_of(elite[0][1])
    print(f"ES {a.task} {tag} gen {g + 1}/{a.gens} best {elite[0][0]:.3f}  sigma_t {np.round(b['sigma_t'], 3).tolist()} "
          f"sigma_cm {np.round(b['sigma_cm'], 3).tolist()}", flush=True)
    json.dump(dict(tag=tag, hist=hist), open(f"{out}/{tag}_hist.json", "w"))

best = elite[0][1]
res = dict(tag=tag, task=a.task, init=a.init, ncal=a.ncal, obj=a.obj, n_evals=len(hist), cal_score=elite[0][0],
           sigma_t=np.exp(best[:T]).tolist(), sigma_cm=np.exp(best[T:]).tolist(), test=[])
for s in range(a.test_seeds):
    res["test"].append(R.run(cfg_of(best, seed=s), data=R.data))
if a.init == "scratch":  # the shipped schedule on the same calibration images, for reference
    res["shipped_cal"] = R.run(ship, data=cal[:a.ncal])
json.dump(res, open(f"{out}/{tag}.json", "w"))
print(f"FINAL {a.task} {tag}: cal {elite[0][0]:.3f} | test PSNR "
      f"{np.mean([mean(r, 'psnr') for r in res['test']]):.2f} LPIPS {np.mean([mean(r, 'lpips') for r in res['test']]):.3f}", flush=True)
