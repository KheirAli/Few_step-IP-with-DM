"""Gate 1 for 'phase retrieval -> partial Fourier reconstruction': give the TRUE phases inside a low-frequency disc (diagnostic only)
and see how good the coarse face becomes.  Also: how accurate are the phases of our current coarse face, by frequency band.  Non-test images."""
import sys, os, glob, re, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
R = torch.load("../mode_probe/reg_best.pt"); W, m_, s_ = R["W"].to(dev), R["m"].to(dev), R["s"].to(dev); OFF = torch.load("../mode_probe/face_bank_8192.pt")["rbar"].to(dev); rs = torch.load("../mode_probe/reg_resstd.pt").to(dev)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); crop = lambda g: g[..., 64:320, 64:320]; Af = lambda im: FU.fft2_m(pad(im)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
a, b = map(int, os.environ["SHARD"].split(":")); files = sorted(glob.glob(os.environ["VAL"] + "/**/*.png", recursive=True), key=nat)[a:b]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
ii = torch.arange(384, device=dev).float() - 192; RAD = (ii[:, None] ** 2 + ii[None, :] ** 2).sqrt()[None, None]
def fit(y, c0, M=None, bO=None):
    c = c0.clone()
    for k in (64, 128, 256, 512):
        ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); G = Af(mu + (ck @ Uf[:k]).view(1, 3, 256, 256)); l = (G.abs() + OFF - y) ** 2
            if M is not None: l = torch.where(M, (G - bO).abs() ** 2, l)
            l = l.mean() + 1e-4 * (ck ** 2 / var[:k]).mean(); l.backward(); return l
        opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
    return c
def soft(x, y, M=None, bO=None, r=1.0):
    G = Af(x); Gn = G / (G.abs() + 1e-8) * (y + r * G.abs()) / (1 + r)
    if M is not None: Gn = torch.where(M, (bO + r * G) / (1 + r), Gn)
    return crop(FU.ifft2_m(Gn).real).clamp(0, 1)
def tv(x, lam=0.05, n=30):
    px = torch.zeros_like(x); py = torch.zeros_like(x)
    for _ in range(n):
        u = x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2)); gx = torch.roll(u, -1, 3) - u; gy = torch.roll(u, -1, 2) - u; nrm = 1 + 0.25 / lam * (gx ** 2 + gy ** 2).sqrt()
        px = (px + 0.25 / lam * gx) / nrm; py = (py + 0.25 / lam * gy) / nrm
    return (x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2))).clamp(0, 1)
def loop(x0, y, M=None, bO=None):
    xx = x0.clone()
    for _ in range(40): xx = tv(soft(xx, y, M, bO))
    return xx
BANDS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 192)]
def phacc(xh, Gt):                                            # magnitude-weighted cos of the phase error, per band (1 = perfect, 0 = random)
    Gh = Af(xh); cs = torch.cos(Gh.angle() - Gt.angle()); w = Gt.abs(); return [float((cs * w)[((RAD >= lo) & (RAD < hi)).expand_as(w)].sum() / w[((RAD >= lo) & (RAD < hi)).expand_as(w)].sum()) for lo, hi in BANDS]
out = []
for fi, f in enumerate(files):
    torch.manual_seed(5000 + a + fi); x = ld(f); Gt = Af(x); y = Gt.abs() + 0.05 * torch.randn(1, 3, 384, 384, device=dev); rec = {}
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; best = None
    for j in range(4):
        c = fit(y, creg + (0 if j == 0 else 1.0) * torch.randn_like(creg) * rs); l = float(((Af(mu + (c @ Uf).view(1, 3, 256, 256)).abs() + OFF - y) ** 2).mean())
        if best is None or l < best[0]: best = (l, c)
    xf = (mu + (best[1] @ Uf).view(1, 3, 256, 256)).clamp(0, 1); xc = loop(xf, y); rec["current"] = ps(xc, x)
    rec["acc_current"] = phacc(xc, Gt); rec["acc_reg"] = phacc((mu + (creg @ Uf).view(1, 3, 256, 256)).clamp(0, 1), Gt); rec["acc_mean"] = phacc(mu.clamp(0, 1), Gt)
    for r_ in (2, 4, 8, 16, 32, 64):
        M = (RAD < r_).expand(1, 3, 384, 384); bO = y * torch.exp(1j * Gt.angle())            # noisy magnitude, true phase, inside the disc
        rec[f"zf_{r_}"] = ps(crop(FU.ifft2_m(torch.where(M, bO, torch.zeros_like(bO))).real).clamp(0, 1), x)
        rec[f"lin_{r_}"] = ps(crop(FU.ifft2_m(torch.where(M, Gt, torch.zeros_like(Gt))).real).clamp(0, 1), x)   # ceiling of the band itself (true coefficients)
        co = fit(y, creg, M, bO); xo = (mu + (co @ Uf).view(1, 3, 256, 256)).clamp(0, 1); rec[f"fit_{r_}"] = ps(xo, x); rec[f"loop_{r_}"] = ps(loop(xo, y, M, bO), x)
        rec[f"cur_loop_{r_}"] = ps(loop(xc, y, M, bO), x)                                     # current coarse face, then loop with the oracle band
    out.append(rec)
json.dump(out, open(f"../mode_probe/logs_x/oph_{a}.json", "w"))
