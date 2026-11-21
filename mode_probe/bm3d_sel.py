"""Choose the BM3D settings for the plug-and-play coarse stage of unmasked phase retrieval, on images that are NOT in the test set.
Start: measured-only eigenface fit with the constant offset (one start).  Loop: noise-aware magnitude fit, then the denoiser."""
import sys, os, glob, re, time, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); sys.path.append("/usr/scratch/akheirandish3/PNP/pylibs_bm3d"); import functions.fft_util as FU, bm3d
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
R = torch.load("../mode_probe/reg_best.pt"); W, m_, s_ = R["W"].to(dev), R["m"].to(dev), R["s"].to(dev); OFF = torch.load("../mode_probe/face_bank_8192.pt")["rbar"].to(dev)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); crop = lambda g: g[..., 64:320, 64:320]; Af = lambda im: FU.fft2_m(pad(im)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
a, b = map(int, os.environ.get("SHARD", "0:24").split(":")); files = sorted(glob.glob(os.environ["VAL"] + "/**/*.png", recursive=True), key=nat)[a:b]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
def fit(y, c0):
    c = c0.clone()
    for k in (64, 128, 256, 512):
        ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = ((Af(mu + (ck @ Uf[:k]).view(1, 3, 256, 256)).abs() + OFF - y) ** 2).mean() + 1e-4 * (ck ** 2 / var[:k]).mean(); l.backward(); return l
        opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
    return c
def soft(x, y, r=1.0):
    G = Af(x); return crop(FU.ifft2_m(G / (G.abs() + 1e-8) * (y + r * G.abs()) / (1 + r)).real).clamp(0, 1)
def tv(x, lam=0.05, n=30):
    px = torch.zeros_like(x); py = torch.zeros_like(x)
    for _ in range(n):
        u = x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2)); gx = torch.roll(u, -1, 3) - u; gy = torch.roll(u, -1, 2) - u; nrm = 1 + 0.25 / lam * (gx ** 2 + gy ** 2).sqrt()
        px = (px + 0.25 / lam * gx) / nrm; py = (py + 0.25 / lam * gy) / nrm
    return (x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2))).clamp(0, 1)
def bm(x, sig, al=1.0):
    d = torch.from_numpy(np.asarray(bm3d.bm3d_rgb(x[0].permute(1, 2, 0).double().cpu().numpy(), sigma_psd=sig), dtype=np.float32)).permute(2, 0, 1)[None].to(dev)
    return ((1 - al) * x + al * d).clamp(0, 1)
res = {}; add = lambda k_, v: res.setdefault(k_, []).append(v)
for f in files:
    torch.manual_seed(1000 + nat(os.path.basename(f))[1] if len(nat(os.path.basename(f))) > 1 else 0)
    x = ld(f); y = Af(x).abs() + 0.05 * torch.randn(1, 3, 384, 384, device=dev)
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; xm = (mu + (fit(y, creg) @ Uf).view(1, 3, 256, 256)).clamp(0, 1); add("start", ps(xm, x))
    def loop(name, den, n, marks):
        xx = xm.clone(); t = time.time()
        for it in range(n):
            xx = den(soft(xx, y), it)
            if it + 1 in marks: add(f"{name} | {it + 1} its", ps(xx, x)); add(f"{name} | {it + 1} its | sec", time.time() - t)
        return xx
    xt = loop("TV 0.05 (current)", lambda v, it: tv(v), 40, (40,))
    for sg in (0.03, 0.06, 0.12, 0.2): loop(f"BM3D sigma {sg}", lambda v, it, sg=sg: bm(v, sg), 10, (5, 10))
    loop("BM3D sigma 0.06, damped 0.5", lambda v, it: bm(v, 0.06, 0.5), 10, (5, 10))
    loop("BM3D sigma 0.12, damped 0.5", lambda v, it: bm(v, 0.12, 0.5), 10, (5, 10))
    loop("TV 30 then BM3D 0.06 x10", lambda v, it: tv(v) if it < 30 else bm(v, 0.06), 40, (40,))
    loop("TV 30 then BM3D 0.12 x10", lambda v, it: tv(v) if it < 30 else bm(v, 0.12), 40, (40,))
    loop("BM3D 0.12 x5 then TV 35", lambda v, it: bm(v, 0.12) if it < 5 else tv(v), 40, (40,))
    add("TV 40, then one BM3D 0.06 (no loop)", ps(bm(xt, 0.06), x))
json.dump(res, open(f"../mode_probe/logs_x/bm3dsel_{a}.json", "w"))
