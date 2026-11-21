"""Unmasked phase retrieval: is a good candidate already among the starts, and can a cheap score pick it?  Non-test images.
8 starts per image (start 0 = regression prediction, others = prediction + residual-sized noise), each: eigenface fit with offset, then TV loop."""
import sys, os, glob, re, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
R = torch.load("../mode_probe/reg_best.pt"); W, m_, s_ = R["W"].to(dev), R["m"].to(dev), R["s"].to(dev); OFF = torch.load("../mode_probe/face_bank_8192.pt")["rbar"].to(dev); rs = torch.load("../mode_probe/reg_resstd.pt").to(dev)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); crop = lambda g: g[..., 64:320, 64:320]; Af = lambda im: FU.fft2_m(pad(im)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
a, b = map(int, os.environ["SHARD"].split(":")); files = sorted(glob.glob(os.environ["VAL"] + "/**/*.png", recursive=True), key=nat)[a:b]
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
out = []
for fi, f in enumerate(files):
    torch.manual_seed(5000 + a + fi); x = ld(f); y = Af(x).abs() + 0.05 * torch.randn(1, 3, 384, 384, device=dev)
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; cands = []
    for j in range(8):
        c = fit(y, creg + (0 if j == 0 else 1.0) * torch.randn_like(creg) * rs); xf = (mu + (c @ Uf).view(1, 3, 256, 256)).clamp(0, 1); xt = xf.clone()
        for _ in range(40): xt = tv(soft(xt, y))
        gx = xt - torch.roll(xt, 1, 3); gy = xt - torch.roll(xt, 1, 2)
        cands.append(dict(psnr_fit=ps(xf, x), psnr=ps(xt, x), psnr_flip=ps(torch.flip(xt, dims=(2, 3)), x), loss_fit=float(((Af(xf).abs() + OFF - y) ** 2).mean()), loss_tv=float(((Af(xt).abs() + OFF - y) ** 2).mean()),
                          coefprior=float((c ** 2 / var).mean()), tvnorm=float((gx.abs() + gy.abs()).mean()), tvres=float(((xt - tv(xt, 0.1)) ** 2).mean()), dmean=float(((xt - mu.clamp(0, 1)) ** 2).mean()), dreg=float((((c - creg) / rs) ** 2).mean())))
    out.append(cands)
json.dump(out, open(f"../mode_probe/logs_x/cand_{a}.json", "w"))
