"""Fit the linear-extrapolation coefficients of step K for one task from a selection run (least squares against the truth).
usage: lle_fit.py <exp pattern> <coef json> <step K> [prmask]"""
import sys, glob, re, json, itertools, numpy as np, torch
from PIL import Image
pat, cf, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); pr = len(sys.argv) > 4; F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
G = None; b = None
for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
    for rf in glob.glob(d + "/raw_*.pt"):
        i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); R = torch.load(rf)
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; S = {k: v[0].float() for _, k, v in R["steps"]}
        if pr: gt = gt * torch.sign((R["x_raw"][0].float() * gt).sum(dim=(1, 2), keepdim=True))
        H = torch.stack([S[j] for j in range(K)] + [S[100 + K]]).flatten(1).double()          # earlier passed-on images, then this step's raw output
        g = H @ H.T; r = H @ gt.flatten().double(); G = g if G is None else G + g; b = r if b is None else b + r
c = torch.linalg.solve(G + 1e-6 * torch.eye(K + 1, dtype=torch.float64) * G.diagonal().mean(), b).tolist()
C = json.load(open(cf)) if glob.glob(cf) else {}; C[str(K)] = c; json.dump(C, open(cf, "w")); print(f"LLEFIT {pat} step {K + 1}: coefficients " + " ".join(f"{v:+.3f}" for v in c) + f" (sum {sum(c):.3f})")
