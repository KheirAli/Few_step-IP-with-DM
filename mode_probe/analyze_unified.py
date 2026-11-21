"""Evaluate unified.py runs.  Usage (from PnP-CM root):
   python ../mode_probe/analyze_unified.py <prefix> <tag> [<tag> ...]      e.g.  cal a b c d   |   fin base uni
Reads exp_<prefix>_<tag>_<task>_s*/ .  PSNR for phase retrieval is with the sign known (best of 8 flips).
Timing excludes the first image of every shard (GPU warm-up)."""
import sys, glob, itertools, json, re
import numpy as np, torch, lpips
from PIL import Image

prefix, tags = sys.argv[1], sys.argv[2:]
TASKS = ["sr_bicubic", "deblur_gauss", "inpainting", "deblur_nl", "jpeg", "phase_retrieval"]
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
torch.set_num_threads(4)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)


def score(x, gt, pr):
    if x is None:
        return float("nan"), float("nan")
    c = (F * x) if pr else x
    ps = [float(10 * torch.log10(1 / (((k + 1) / 2 - (gt + 1) / 2) ** 2).mean())) for k in c]
    k = int(np.argmax(ps))
    with torch.no_grad():
        l = float(lp(c[k:k + 1].to(dev), gt[None].to(dev)))
    return ps[k], l


out = {}
for task in TASKS:
    pr = task == "phase_retrieval"
    for tag in tags:
        rows = []
        for d in sorted(glob.glob(f"../mode_probe/exp_{prefix}_{tag}_{task}_s*")):
            if not glob.glob(f"{d}/stats.json"):
                continue
            st = json.load(open(f"{d}/stats.json")); first_img = min(x["img"] for x in st)
            for x in st:
                i = x["img"]; lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
                if not lab or not glob.glob(f"{d}/raw_{i}.pt"):
                    continue
                gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
                raw = torch.load(f"{d}/raw_{i}.pt")
                p2, l2 = score(raw["x_raw"], gt, pr); p1, l1 = score(raw.get("x_first"), gt, pr)
                rows.append(dict(x, p_out=p2, l_out=l2, p_first=p1, l_first=l1, warm=(i != first_img)))
        if not rows:
            continue
        m = lambda k, rr=rows: float(np.mean([r[k] for r in rr if k in r and r[k] == r[k]])) if any(k in r for r in rr) else float("nan")
        se = lambda k: float(np.std([r[k] for r in rows]) / np.sqrt(len(rows)))
        tw = [r for r in rows if r["warm"]]
        line = f"{task:16s} {tag:5s} n={len(rows):3d} | out PSNR {m('p_out'):5.2f}±{se('p_out'):.2f} LPIPS {m('l_out'):.3f}"
        if "first" in rows[0]:
            f = lambda k: float(np.mean([r["first"][k] for r in rows]))
            line += (f" | pass1 PSNR {m('p_first'):5.2f} LPIPS {m('l_first'):.3f} cost A {f('A'):5.0f} At {f('At'):5.0f} jvp {f('jvp'):5.0f}"
                     f" | move(z1) {m('move'):.3f} [{min(r['move'] for r in rows):.3f},{max(r['move'] for r in rows):.3f}]"
                     f" move(z0) {m('move0'):.3f} [{min(r['move0'] for r in rows):.3f},{max(r['move0'] for r in rows):.3f}] sigma1 {m('sigma1'):.3f}"
                     f" | 2nd pass {sum(r['second_pass'] for r in rows)}/{len(rows)}")
        line += f" | total: time {m('time', tw):6.2f}s NFE {m('nfe'):4.1f} A {m('A'):6.0f} At {m('At'):6.0f} jvp {m('jvp'):5.0f}"
        print(line, flush=True)
        out[f"{task}/{tag}"] = rows
json.dump(out, open(f"../mode_probe/unified_{prefix}_results.json", "w"))
