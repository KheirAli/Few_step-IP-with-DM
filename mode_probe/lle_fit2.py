"""Fit step K of the linear extrapolation with weights that vary over the image.
mode: scalar (one weight per image, as before) | pix (one per pixel and colour) | pixs (per pixel, shared by the colours) | blkB (constant on BxB blocks, shared colours)
Per location the weights solve a small least-squares problem over the calibration images, with a ridge pulling them to the scalar solution.
usage: lle_fit2.py <exp pattern> <out .pt> <K> <mode> [pr]"""
import sys, glob, re, os, math, itertools, numpy as np, torch
from PIL import Image
pat, out, K, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; pr = len(sys.argv) > 5; dev = torch.device("cuda:0"); lam = float(os.environ.get("LLE_RIDGE", "0.05"))
# structured modes: band<B> (per radial frequency band) | gate (per level of step-to-step change, nonlinear) | zscalar / zband<B> (data-fit image added as a term)
# zbandH<B>: band weights with ALL data-fit images so far as terms.   zreg<B>: band weights fitted separately on the well-measured pixels and on the rest.
ZALL = mode.startswith("zbandH"); REG = mode.startswith("zreg")
if ZALL: mode_ = "zband" + mode[6:]
else: mode_ = mode
ZF = mode.startswith("z"); kind = ("reg" if REG else "band" if "band" in mode else "gate" if mode == "gate" else "scalar" if mode == "zscalar" else ""); STRUCT = kind != ""
if STRUCT:
    sys.path.insert(0, "../mode_probe"); THR = torch.tensor([0.01, 0.02, 0.04, 0.08, 0.16]); NB = int(mode_.split("band")[1]) if kind == "band" else int(mode[4:]) if kind == "reg" else (6 if kind == "gate" else 1); T_ = K + 1 + (K + 1 if ZALL else int(ZF))
    if kind == "reg": Gr = torch.zeros(2, T_ * NB, T_ * NB, device=dev, dtype=torch.float64); br = torch.zeros(2, T_ * NB, device=dev, dtype=torch.float64)
    Gq = torch.zeros(NB, T_, T_, device=dev, dtype=torch.float64); bq = torch.zeros(NB, T_, device=dev, dtype=torch.float64)
    def band_index(nn_, B):
        f = torch.fft.fftfreq(nn_, device=dev) * nn_; r = (f[:, None] ** 2 + f[None, :] ** 2).sqrt(); return torch.bucketize(r, torch.tensor([1.5 * 2 ** k for k in range(B - 1)], device=dev))
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
        if STRUCT:
            Hs = torch.stack([S[j] for j in range(K)] + ([S[200 + j] for j in range(K + 1)] if ZALL else [S[200 + K]] if ZF else []) + [S[100 + K]]).to(dev).double()
            if kind == "reg":
                Mk = (S[300].to(dev).double() if 300 in S else torch.ones(3, 256, 256, device=dev, dtype=torch.float64)); bidx = band_index(256, NB); Fh = torch.fft.fft2(Hs, norm="ortho")
                Cc = torch.stack([torch.fft.ifft2(Fh[j] * (bidx == b_)[None], norm="ortho").real for j in range(T_) for b_ in range(NB)]).flatten(1)        # [T*NB, pixels]
                for r_, Mm in enumerate((Mk.flatten(), 1 - Mk.flatten())):
                    Cm = Cc * Mm; Gr[r_] += Cm @ Cc.T; br[r_] += Cm @ gt.flatten()
                n += 1; continue
            if kind == "band": Hq = torch.fft.fft2(Hs, norm="ortho"); gq = torch.fft.fft2(gt, norm="ortho"); bi = band_index(256, NB)[None].expand(3, 256, 256).reshape(-1)
            elif kind == "gate": Hq = Hs; gq = gt; bi = torch.bucketize((Hs[-1] - Hs[K - 1]).abs().mean(0), THR.to(dev).double())[None].expand(3, 256, 256).reshape(-1)
            else: Hq = Hs; gq = gt; bi = torch.zeros(3 * 256 * 256, dtype=torch.long, device=dev)
            Hq = Hq.flatten(1); gq = gq.flatten()
            for j in range(T_):
                bq[:, j].index_add_(0, bi, (Hq[j].conj() * gq).real)
                for k in range(j, T_): v_ = torch.zeros(NB, device=dev, dtype=torch.float64).index_add_(0, bi, (Hq[j].conj() * Hq[k]).real); Gq[:, j, k] += v_; Gq[:, k, j] += v_ * (k != j)
            n += 1; continue
        if EIG:
            Hf = H.flatten(1); co = Hf @ Ub.T; g_ = gt.flatten() @ Ub.T; R_ = Hf - co @ Ub; rg = gt.flatten() - g_ @ Ub
            Gm += co.T[:, :, None] * co.T[:, None, :]; bm += co.T * g_[:, None]; Gr += R_ @ R_.T; br += R_ @ rg; n += 1; continue
        G += H[:, None] * H[None]; b += H * gt[None]; n += 1
if EIG:
    I_ = torch.eye(K + 1, device=dev, dtype=torch.float64); sr = torch.linalg.solve(Gr + 1e-6 * Gr.diagonal().mean() * I_, br)
    A_ = torch.linalg.solve(Gm + lam * Gm.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * I_, (bm + lam * Gm.diagonal(dim1=1, dim2=2).mean(1)[:, None] * sr)[..., None])[..., 0].T    # [K+1, 30]
    C = torch.load(out) if os.path.exists(out) else {}; C[str(K)] = {"U": Ub.float().cpu(), "A": A_.float().cpu(), "s": sr.float().cpu()}; torch.save(C, out)
    print(f"LLEFIT2 {pat} step {K + 1} mode {mode} n={n}: weight outside the embedding " + " ".join(f"{v:+.3f}" for v in sr.tolist()) + f" | last-step weight across the 30 coordinates: min {float(A_[-1].min()):+.2f} max {float(A_[-1].max()):+.2f}"); sys.exit(0)
if STRUCT and kind == "reg":
    I_ = torch.eye(T_ * NB, device=dev, dtype=torch.float64); Gall = Gr.sum(0); wall = torch.linalg.pinv(Gall + 1e-6 * Gall.diagonal().mean() * I_, hermitian=True, rtol=1e-9) @ br.sum(0); Wr = []
    for r_ in range(2):
        tr_ = Gr[r_].diagonal().mean(); Wr.append(wall if float(tr_) < 1e-9 * float(Gall.diagonal().mean()) else torch.linalg.pinv(Gr[r_] + lam * tr_ * I_, hermitian=True, rtol=1e-9) @ (br[r_] + lam * tr_ * wall))
    Wr = torch.stack(Wr).view(2, T_, NB); C = torch.load(out) if os.path.exists(out) else {}; C[str(K)] = {"kind": "regband", "z": True, "W": Wr.float().cpu()}; torch.save(C, out)
    print(f"LLEFIT2 {pat} step {K + 1} mode {mode} n={n}: last term, well-measured region per band " + " ".join(f"{v:+.2f}" for v in Wr[0, -1].tolist()) + " | other region " + " ".join(f"{v:+.2f}" for v in Wr[1, -1].tolist())); sys.exit(0)
if STRUCT:
    # pseudo-inverse: two terms can coincide exactly (e.g. a data-fit image equal to the previous passed-on image), and a bin can be empty
    I_ = torch.eye(T_, device=dev, dtype=torch.float64); Gs_ = Gq.sum(0); w0 = torch.linalg.pinv(Gs_ + 1e-6 * Gs_.diagonal().mean() * I_, hermitian=True, rtol=1e-9) @ bq.sum(0); tr = Gq.diagonal(dim1=1, dim2=2).mean(1)
    tr = torch.where(tr > 1e-9 * tr.max(), tr, tr.max())
    Wq = (torch.linalg.pinv(Gq + lam * tr[:, None, None] * I_, hermitian=True, rtol=1e-9) @ (bq + lam * tr[:, None] * w0)[..., None])[..., 0].T          # [terms, bins]
    if kind == "scalar": Wq = w0[:, None].expand(T_, NB).clone()
    C = torch.load(out) if os.path.exists(out) else {}; C[str(K)] = {"kind": "gate" if kind == "gate" else "band", "z": ZF, "zall": ZALL, "W": Wq.float().cpu(), "thr": THR.float().cpu()}; torch.save(C, out)
    print(f"LLEFIT2 {pat} step {K + 1} mode {mode} n={n}: overall " + " ".join(f"{v:+.3f}" for v in w0.tolist()) + " | last term per bin " + " ".join(f"{v:+.2f}" for v in Wq[-1].tolist()) + (" | data-fit term per bin " + " ".join(f"{v:+.2f}" for v in Wq[-2].tolist()) if ZF else "")); sys.exit(0)
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
