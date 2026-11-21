"""Fit step K of the linear extrapolation with weights that vary over the image.
mode: scalar (one weight per image, as before) | pix (one per pixel and colour) | pixs (per pixel, shared by the colours) | blkB (constant on BxB blocks, shared colours)
Per location the weights solve a small least-squares problem over the calibration images, with a ridge pulling them to the scalar solution.
usage: lle_fit2.py <exp pattern> <out .pt> <K> <mode> [pr]"""
import sys, glob, re, os, math, itertools, numpy as np, torch
from PIL import Image
pat, out, K, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; pr = len(sys.argv) > 5; dev = torch.device("cuda:0"); lam = float(os.environ.get("LLE_RIDGE", "0.05"))
EIG = mode.startswith("eig")
if EIG:
    Ub = torch.load("../mode_probe/lle/eig30_basis.pt").to(dev).double(); Gm = torch.zeros(30, K + 1, K + 1, device=dev, dtype=torch.float64); bm = torch.zeros(30, K + 1, device=dev, dtype=torch.float64)
    Gr = torch.zeros(K + 1, K + 1, device=dev, dtype=torch.float64); br = torch.zeros(K + 1, device=dev, dtype=torch.float64)
G = torch.zeros(K + 1, K + 1, 3, 256, 256, device=dev, dtype=torch.float64); b = torch.zeros(K + 1, 3, 256, 256, device=dev, dtype=torch.float64); n = 0
for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
    for rf in glob.glob(d + "/raw_*.pt"):
        i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); R = torch.load(rf)
        gt = (torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1).to(dev).double(); S = {k: v[0].float() for _, k, v in R["steps"]}
        if pr: gt = gt * torch.sign((R["x_raw"][0].to(dev).double() * gt).sum(dim=(1, 2), keepdim=True))
        H = torch.stack([S[j] for j in range(K)] + [S[100 + K]]).to(dev).double()          # [K+1, 3, 256, 256]
        if EIG:
            Hf = H.flatten(1); co = Hf @ Ub.T; g_ = gt.flatten() @ Ub.T; R_ = Hf - co @ Ub; rg = gt.flatten() - g_ @ Ub
            Gm += co.T[:, :, None] * co.T[:, None, :]; bm += co.T * g_[:, None]; Gr += R_ @ R_.T; br += R_ @ rg; n += 1; continue
        G += H[:, None] * H[None]; b += H * gt[None]; n += 1
if EIG:
    I_ = torch.eye(K + 1, device=dev, dtype=torch.float64); sr = torch.linalg.solve(Gr + 1e-6 * Gr.diagonal().mean() * I_, br)
    A_ = torch.linalg.solve(Gm + lam * Gm.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * I_, (bm + lam * Gm.diagonal(dim1=1, dim2=2).mean(1)[:, None] * sr)[..., None])[..., 0].T    # [K+1, 30]
    C = torch.load(out) if os.path.exists(out) else {}; C[str(K)] = {"U": Ub.float().cpu(), "A": A_.float().cpu(), "s": sr.float().cpu()}; torch.save(C, out)
    print(f"LLEFIT2 {pat} step {K + 1} mode {mode} n={n}: weight outside the embedding " + " ".join(f"{v:+.3f}" for v in sr.tolist()) + f" | last-step weight across the 30 coordinates: min {float(A_[-1].min()):+.2f} max {float(A_[-1].max()):+.2f}"); sys.exit(0)
Gs = G.sum(dim=(2, 3, 4)); bs = b.sum(dim=(1, 2, 3)); w0 = torch.linalg.solve(Gs + 1e-6 * torch.eye(K + 1, device=dev, dtype=torch.float64) * Gs.diagonal().mean(), bs)
if mode == "scalar":
    W = w0.view(K + 1, 1, 1, 1).expand(K + 1, 3, 256, 256)
elif mode.startswith("dct"):
    # weight maps restricted to the M smoothest 2-D cosine patterns: (K+1) x M numbers instead of one per pixel
    M = int(mode[3:]); xs = (torch.arange(256, device=dev, dtype=torch.float64) + 0.5) / 256; idx = sorted(itertools.product(range(8), range(8)), key=lambda t: (t[0] + t[1], t[0]))[:M]
    Phi = torch.stack([(torch.cos(math.pi * u * xs)[:, None] * torch.cos(math.pi * v * xs)[None, :]).flatten() for u, v in idx]); Phi = Phi / Phi.norm(dim=1, keepdim=True) * 256
    Gp = G.sum(2).flatten(2); bp = b.sum(1).flatten(1); Gb = torch.einsum("mp,np,jkp->jmkn", Phi, Phi, Gp).reshape((K + 1) * M, (K + 1) * M); bb = torch.einsum("mp,jp->jm", Phi, bp).reshape(-1)
    a0 = torch.zeros(K + 1, M, device=dev, dtype=torch.float64); a0[:, 0] = w0 / float(Phi[0, 0]); reg = lam * Gb.diagonal().mean()
    a_ = torch.linalg.solve(Gb + reg * torch.eye(Gb.shape[0], device=dev, dtype=torch.float64), bb + reg * a0.reshape(-1)).view(K + 1, M)
    W = (a_ @ Phi).view(K + 1, 1, 256, 256).expand(K + 1, 3, 256, 256)
else:
    Gm, bm = G, b
    if mode == "pixs" or mode.startswith("blk"): Gm = Gm.sum(2, keepdim=True); bm = bm.sum(1, keepdim=True)
    if mode.startswith("blk"):
        B = int(mode[3:]); pool = lambda t: torch.nn.functional.avg_pool2d(t.flatten(0, -3), B).view(*t.shape[:-2], 256 // B, 256 // B); Gm = pool(Gm); bm = pool(bm)
    A = Gm.permute(2, 3, 4, 0, 1); r = bm.permute(1, 2, 3, 0); tr = A.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
    Wl = torch.linalg.solve(A + lam * tr * torch.eye(K + 1, device=dev, dtype=torch.float64), (r + lam * tr[..., 0] * w0)[..., None])[..., 0].permute(3, 0, 1, 2)   # [K+1, C', h, w]
    if mode.startswith("blk"): Wl = torch.nn.functional.interpolate(Wl, size=256, mode="bilinear", align_corners=False)
    W = Wl.expand(K + 1, 3, 256, 256)
C = torch.load(out) if os.path.exists(out) else {}; C[str(K)] = W.float().cpu().clone(); torch.save(C, out)
print(f"LLEFIT2 {pat} step {K + 1} mode {mode} n={n}: scalar solution " + " ".join(f"{v:+.3f}" for v in w0.tolist()) + f" | spatial std of the last weight {float(W[-1].std()):.3f}")
