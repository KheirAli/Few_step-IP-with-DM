"""Merge sharded run_fast.py outputs (exp_<variant>_s<start>/) and report quality + cost.

Phase retrieval: PSNR/LPIPS with the sign known (best of the 8 per-channel flips against ground
truth) and with the repo's rule (flip closest in LPIPS to the fixed reference face).
Other tasks: plain PSNR/LPIPS.

Run from the PnP-CM repo root:  python ../mode_probe/analyze_shards.py <pr|other> variant [variant ...]
"""
import sys, glob, itertools, json, re
import numpy as np, torch, lpips, scipy.io as sio
from PIL import Image

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
kind, variants = sys.argv[1], sys.argv[2:]
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3)), device=dev).view(8, 3, 1, 1)
ref = torch.from_numpy(sio.loadmat("exp/phase_retrieval_ref_img/cm_out.mat")["cm_out"]).float().to(dev)


def psnr(a, b):
    return float(10 * torch.log10(1 / (((a + 1) / 2 - (b + 1) / 2) ** 2).mean()))


out = {}
for v in variants:
    rows = {}
    stats = {}
    for d in sorted(glob.glob(f"../mode_probe/exp_{v}_s*")):
        for st in json.load(open(f"{d}/stats.json")) if glob.glob(f"{d}/stats.json") else []:
            stats[st["img"]] = st
        for rf in glob.glob(f"{d}/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0])
            lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab:
                continue
            x = torch.load(rf)["x_raw"].to(dev)
            gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1).to(dev) * 2 - 1
            with torch.no_grad():
                if kind == "pr":
                    c = F * x
                    ps = [psnr(c[k], gt) for k in range(8)]
                    lg = lp(c, gt[None].expand(8, -1, -1, -1)).flatten()
                    kb = int(np.argmax(ps)); kr = int(lp(c, ref.expand(8, -1, -1, -1)).flatten().argmin())
                    rows[i] = dict(psnr_known=ps[kb], lpips_known=float(lg[kb]), psnr_ref=ps[kr], lpips_ref=float(lg[kr]),
                                   raw_ok=(kb == 0), ref_ok=(kr == kb))
                else:
                    rows[i] = dict(psnr=psnr(x[0], gt), lpips=float(lp(x, gt[None])))
    n = len(rows)
    if n == 0:
        print(f"{v:26s} no data"); continue
    m = lambda k: float(np.mean([r[k] for r in rows.values()]))
    sd = lambda k: float(np.std([r[k] for r in rows.values()]) / np.sqrt(n))
    c = lambda k: float(np.mean([stats[i][k] for i in rows if i in stats])) if stats else float("nan")
    cost = f"inner {c('t_inner'):6.2f}s  fwd {c('fwd'):5.0f} grad {c('vjp'):5.0f} jvp {c('jvp'):5.0f}"
    if kind == "pr":
        fail = sum(r["psnr_known"] < 18 for r in rows.values())
        print(f"{v:26s} n={n:3d} | sign known: PSNR {m('psnr_known'):5.2f}±{sd('psnr_known'):.2f} LPIPS {m('lpips_known'):.3f} (<18dB: {fail:2d}) | "
              f"repo fix: PSNR {m('psnr_ref'):5.2f} LPIPS {m('lpips_ref'):.3f} sign ok {sum(r['ref_ok'] for r in rows.values()):3d}/{n} | "
              f"raw sign ok {sum(r['raw_ok'] for r in rows.values()):3d} | {cost}")
    else:
        print(f"{v:26s} n={n:3d} | PSNR {m('psnr'):5.2f}±{sd('psnr'):.2f} LPIPS {m('lpips'):.3f}±{sd('lpips'):.3f} min {min(r['psnr'] for r in rows.values()):.1f} | {cost}")
    out[v] = {"rows": rows, "n": n}
json.dump(out, open(f"../mode_probe/shards_{kind}_results.json", "w"))
