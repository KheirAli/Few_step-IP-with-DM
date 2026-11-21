"""Does the spread between sampler runs show WHERE the error is?  (no GPU)
For each image: per-pixel variance across N runs (s2) vs per-pixel squared error of the run average (e2), on 8x8 blocks.
  ratio    = mean e2 / mean s2     (a calibrated posterior sampler gives 1 + 1/N)
  corr     = rank correlation between the block maps of s2 and e2
  top20    = share of the total error that lies in the 20% of blocks with the largest spread (0.20 = no information)
  python spread_vs_error.py v1,v2,v3,v4"""
import glob, itertools, re, sys
import numpy as np, torch
from PIL import Image
import torch.nn.functional as Fn
torch.set_num_threads(8)
vs = sys.argv[1].split(","); N = len(vs)
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
def rank(a): return torch.argsort(torch.argsort(a)).float()
for t in ["sr_bicubic", "deblur_gauss", "inpainting", "deblur_nl", "jpeg", "phase_retrieval"]:
    pr = t == "phase_retrieval"; L = []
    for v in vs:
        o = {}
        for rf in glob.glob(f"exp_fin_{v}_{t}_s*/raw_*.pt"):
            o[int(re.findall(r"raw_(\d+)\.pt", rf)[0])] = rf
        L.append(o)
    ids = sorted(set.intersection(*[set(o) for o in L])); R, C, T20 = [], [], []
    for i in ids:
        lab = glob.glob(f"exp_fin_{vs[0]}_{t}_s*/image_samples/*/labels/orig_{i}.png")
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
        xs = [torch.load(o[i])["x_raw"][0].float() for o in L]
        if pr:
            xs = [xs[0]] + [x * torch.sign((xs[0] * x).sum(dim=(1, 2), keepdim=True)) for x in xs[1:]]
            k = int(torch.argmin(((F * torch.stack(xs).mean(0) - gt) ** 2).mean(dim=(1, 2, 3)))); xs = [F[k] * x for x in xs]
        X = torch.stack(xs); s2 = X.var(0).mean(0); e2 = ((X.mean(0) - gt) ** 2).mean(0)
        bs, be = [Fn.avg_pool2d(m[None, None], 8).flatten() for m in (s2, e2)]
        R.append(float(e2.mean() / s2.mean())); C.append(float(torch.corrcoef(torch.stack([rank(bs), rank(be)]))[0, 1]))
        top = torch.argsort(bs, descending=True)[: len(bs) // 5]; T20.append(float(be[top].sum() / be.sum()))
    print(f"{t:16s} n={len(ids):3d} | error/spread ratio {np.median(R):5.1f} (calibrated {1 + 1 / N:.2f}) | rank corr {np.mean(C):.2f} | error in top-20% spread blocks {np.mean(T20):.2f}", flush=True)
