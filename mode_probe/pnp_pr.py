"""Plug-and-play coarse stage for unmasked phase retrieval with a CLASSICAL denoiser in the loop (run from PnP-CM root).
Start: measured-only eigenface fit.  Each iteration: noise-aware magnitude fit (closed form), then a total-variation denoiser.
(BM3D is not installed here and would be far too slow on CPU for this many iterations; total variation is the classical prior used.)
Compared with the same loop without the denoiser."""
import sys, os, glob, re, numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
R = torch.load("../mode_probe/reg_best.pt"); W, m_, s_ = R["W"].to(dev), R["m"].to(dev), R["s"].to(dev)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); crop = lambda g: g[..., 64:320, 64:320]; Af = lambda im: FU.fft2_m(pad(im)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
files = sorted(glob.glob("../mode_probe/data_paper/celeba_hq/celeba_hq/*.png"), key=nat)[:int(os.environ.get("NIMG", "24"))]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
def fit(y, c0):
    c = c0.clone()
    for k in (64, 128, 256, 512):
        ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = ((Af(mu + (ck @ Uf[:k]).view(1, 3, 256, 256)).abs() - y) ** 2).mean() + 1e-4 * (ck ** 2 / var[:k]).mean(); l.backward(); return l
        opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
    return c
def soft(x, y, r):                                             # exact per-coefficient minimiser of (|c|-y)^2/2 + r|c-u|^2/2, phase kept
    G = Af(x); return crop(FU.ifft2_m(G / (G.abs() + 1e-8) * (y + r * G.abs()) / (1 + r)).real).clamp(0, 1)
def tv(x, lam, n=30):                                          # Chambolle's total-variation denoiser
    if lam <= 0: return x
    px = torch.zeros_like(x); py = torch.zeros_like(x)
    for _ in range(n):
        div = px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2); u = x + lam * div
        gx = torch.roll(u, -1, 3) - u; gy = torch.roll(u, -1, 2) - u; nrm = 1 + 0.25 / lam * (gx ** 2 + gy ** 2).sqrt()
        px = (px + 0.25 / lam * gx) / nrm; py = (py + 0.25 / lam * gy) / nrm
    return (x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2))).clamp(0, 1)
res = {}; add = lambda k_, v: res.setdefault(k_, []).append(v)
torch.manual_seed(3)
for f in files:
    x = ld(f); y = Af(x).abs() + 0.05 * torch.randn(1, 3, 384, 384, device=dev)
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; xm = (mu + (fit(y, creg) @ Uf).view(1, 3, 256, 256)).clamp(0, 1)
    add("start (measured-only eigenface fit)", ps(xm, x)); add("start, denoised once (TV 0.05)", ps(tv(xm, 0.05), x))
    for r in (1.0, 4.0):
        for lam, lname in ((0.0, "no denoiser"), (0.02, "TV 0.02"), (0.05, "TV 0.05"), (0.1, "TV 0.10"), (-1, "TV 0.10 -> 0.01")):
            xx = xm.clone()
            for it in range(40):
                l_ = lam if lam >= 0 else float(np.geomspace(0.1, 0.01, 40)[it]); xx = tv(soft(xx, y, r), l_)
                if it + 1 in (10, 20, 40): add(f"anchor weight {r:g}, {lname}, {it + 1} iterations", ps(xx, x))
            if lam in (0.0, 0.05): add(f"anchor weight {r:g}, {lname}: magnitude misfit after 40", float((Af(xx).abs() - y).norm() / y.norm()))
print(f"{len(files)} images, PSNR vs the true image before any model call")
for k_, v in res.items(): v = np.array(v); print(f"   {k_:58s} mean {v.mean():7.3f}   5th pct {np.percentile(v, 5):7.3f}   worst {v.min():7.3f}")
