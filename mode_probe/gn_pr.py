"""Full-resolution linearised correction for unmasked phase retrieval (run from PnP-CM root).
Around a current image x0 the magnitudes are linear in a correction d:   |A(x0+d)| ~ |A x0| + Re( conj(u0) * A d ),  u0 = phase of A x0.
Each round solves   min_d  || Re(conj(u0) A d) - (y - |A x0|) ||^2 + lam ||d||^2   by conjugate gradients (FFTs only), with d free at
every pixel (NOT restricted to the eigenface subspace), then clips to [0, 1] and re-linearises.  lam = sigma^2 / tau^2.
Starts: oracle eigenface image (diagnostic), measured-only eigenface fit, and for reference plain projections."""
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
def gn(x, y, lam, rounds, cg=30):
    for _ in range(rounds):
        Z0 = Af(x); u0 = Z0 / (Z0.abs() + 1e-8); b = y - Z0.abs()
        J = lambda d: (u0.conj() * Af(d)).real; Jt = lambda r: crop(FU.ifft2_m(u0 * r.to(torch.complex64)).real)
        H = lambda d: Jt(J(d)) + lam * d
        d = torch.zeros_like(x); r = Jt(b); p = r.clone(); rs = (r * r).sum()
        for _ in range(cg):
            Hp = H(p); a = rs / ((p * Hp).sum() + 1e-30); d = d + a * p; r = r - a * Hp; rn = (r * r).sum()
            if rn < 1e-10 * rs: break
            p = r + (rn / rs) * p; rs = rn
        x = (x + d).clamp(0, 1)
    return x
def er(x, y, k):
    for _ in range(k):
        G = Af(x); x = crop(FU.ifft2_m(y * G / (G.abs() + 1e-8)).real).clamp(0, 1)
    return x
res = {}; add = lambda k_, v: res.setdefault(k_, []).append(v)
torch.manual_seed(3)
for f in files:
    x = ld(f); y = Af(x).abs() + 0.05 * torch.randn(1, 3, 384, 384, device=dev)
    cstar = (x - mu).flatten() @ Uf.T; xo = (mu + (cstar @ Uf).view(1, 3, 256, 256)).clamp(0, 1)
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; xm = (mu + (fit(y, creg) @ Uf).view(1, 3, 256, 256)).clamp(0, 1)
    for tag, x0 in (("ORACLE eigenface start", xo), ("measured-only eigenface start", xm)):
        add(f"{tag}: start", ps(x0, x)); add(f"{tag}: 30 plain projections", ps(er(x0, y, 30), x))
        for tau in (0.02, 0.05, 0.1, 0.3):
            for rounds in (1, 3, 10):
                xx = gn(x0, y, (0.05 / tau) ** 2, rounds); add(f"{tag}: linearised correction, step size {tau}, {rounds} rounds", ps(xx, x))
                if tau == 0.05 and rounds == 3: add(f"{tag}: magnitude misfit after (step 0.05, 3 rounds)", float((Af(xx).abs() - y).norm() / y.norm()))
print(f"{len(files)} images, PSNR vs the true image before any model call")
for k_, v in res.items(): v = np.array(v); print(f"   {k_:76s} mean {v.mean():7.3f}   5th pct {np.percentile(v, 5):7.3f}   worst {v.min():7.3f}")
