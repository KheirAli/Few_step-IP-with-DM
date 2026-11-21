"""Test ground-truth-free rules for 'keep the first pass or the second pass' on v_<task>_one / v_<task>_two.

Run from the PnP-CM repo root:  python ../mode_probe/analyze_rules.py
"""
import glob, itertools, json, re
import numpy as np, torch
from PIL import Image

F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)


def psnr(a, b):
    return float(10 * torch.log10(1 / (((a + 1) / 2 - (b + 1) / 2) ** 2).mean()))


def load(v, pr):
    res = {}
    for d in sorted(glob.glob(f"../mode_probe/exp_{v}_s*")):
        st = {x["img"]: x for x in json.load(open(f"{d}/stats.json"))}
        for rf in glob.glob(f"{d}/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0])
            lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab or i not in st:
                continue
            x = torch.load(rf)["x_raw"]
            gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
            p = max(psnr(c, gt) for c in F * x) if pr else psnr(x[0], gt)
            res[i] = {"psnr": p, **st[i]["diag"], "tau": st[i]["tau"]}
    return res


allrows = {}
for task, pr in [("pr", True), ("db", False), ("jp", False)]:
    one, two = load(f"v_{task}_one", pr), load(f"v_{task}_two", pr)
    ids = sorted(set(one) & set(two))
    if not ids:
        print(task, "no data"); continue
    p1 = np.array([one[i]["psnr"] for i in ids]); p2 = np.array([two[i]["psnr"] for i in ids])
    g = lambda d, k: np.array([d[i][k] for i in ids])
    allrows[task] = dict(ids=ids, p1=p1, p2=p2, move1=g(one, "move_abs"), move2=g(two, "move_abs"),
                         mis1=g(one, "mis_out"), mis2=g(two, "mis_out"), pri1=g(one, "prior_out"), pri2=g(two, "prior_out"),
                         init_ratio=g(one, "mis_init") / g(one, "tau"))
    r = allrows[task]
    print(f"\n== {task}  n={len(ids)}   pass1 {p1.mean():.2f}   pass2 {p2.mean():.2f}   oracle {np.maximum(p1, p2).mean():.2f}   "
          f"pass 2 better on {int((p2 > p1).sum())}/{len(ids)}")
    for name, pick in [("lower data misfit of output", r["mis2"] < r["mis1"]),
                       ("lower prior (denoiser) residual", r["pri2"] < r["pri1"])]:
        print(f"   rule [{name:32s}] keeps pass 2 on {int(pick.sum()):3d}  ->  PSNR {np.where(pick, p2, p1).mean():.2f}")
    q = lambda a: f"min {a.min():.3f} / median {np.median(a):.3f} / max {a.max():.3f}"
    print(f"   how far pass 1 moved from its start (RMS):  {q(r['move1'])}")
    print(f"   how far pass 2 moved from pass 1   (RMS):  {q(r['move2'])}")
    print(f"   start misfit / noise level:                 {q(r['init_ratio'])}")

# one threshold on 'how far pass 1 moved from its start', applied to every task
print("\n== single threshold on pass-1 movement (run pass 2 only if movement > t) ==")
for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    line = f"t={t:.2f}: "
    for task, r in allrows.items():
        pick = r["move1"] > t
        line += f"{task}: 2nd pass on {int(pick.sum()):3d}/{len(pick)} -> {np.where(pick, r['p2'], r['p1']).mean():.2f}   "
    print(line)

# threshold-free rule: the scale is the noise the sampler itself injects at its first step.
# run a second pass only if pass 1 moved further from its start than that noise could cover.
b = torch.linspace(1e-4, 0.02, 1000); ab = torch.cat([torch.ones(1), (1 - b).cumprod(0)])
sig1 = float((1 - ab[252]) ** 0.5)   # iN = 250 for all three tasks -> first-step noise
print(f"\n== noise-level rule: second pass iff pass-1 movement (RMS) > first-step noise sigma = {sig1:.3f} ==")
for task, r in allrows.items():
    pick = r["move1"] > sig1
    print(f"{task}: second pass on {int(pick.sum()):3d}/{len(pick)}  ->  PSNR {np.where(pick, r['p2'], r['p1']).mean():.2f}   "
          f"(always pass 1: {r['p1'].mean():.2f}, always pass 2: {r['p2'].mean():.2f}, oracle: {np.maximum(r['p1'], r['p2']).mean():.2f})")
    if task == "pr":
        skipped = ~pick
        if skipped.any():
            print(f"    skipped images: pass1 {r['p1'][skipped].round(2)} pass2 {r['p2'][skipped].round(2)} move {r['move1'][skipped].round(3)}")
        c = np.corrcoef(r["move1"], r["p2"] - r["p1"])[0, 1]
        print(f"    within phase retrieval: corr(movement, gain from pass 2) = {c:.2f}")
