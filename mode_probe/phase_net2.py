"""Stronger learned predictor for unmasked phase retrieval (option 3).  Input: autocorrelation (the image-domain form of the magnitudes)
+ low-frequency log-magnitudes.  Residual encoder, global bottleneck, decoder to K hypotheses at 64x64.  Trained only on consistency-model
samples (256k).  Tested on real non-test images exactly like phase_net.py."""
import sys, os, glob, re, json, time, numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); K = int(os.environ.get("KH", "8")); EP = int(os.environ.get("EP", "16")); NP = int(os.environ.get("NPARTS", "32")); tag = os.environ.get("TAG2", f"v2k{K}")
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); crop = lambda g: g[..., 64:320, 64:320]; Af = lambda im: FU.fft2_m(pad(im)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
def feat(y):
    yc = y.clamp(min=0); a = FU.ifft2_m((yc ** 2).to(torch.complex64)).real; a = a / a[..., 192:193, 192:193].clamp(min=1e-8)
    return torch.cat([torch.nn.functional.avg_pool2d(a[..., 64:320, 64:320], 2), torch.log1p(yc)[..., 128:256, 128:256]], 1)
low = lambda x: torch.nn.functional.avg_pool2d(x, 4); up = lambda z: torch.nn.functional.interpolate(z, size=256, mode="bicubic", align_corners=False).clamp(0, 1)
class Res(nn.Module):
    def __init__(s, c): super().__init__(); s.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1))
    def forward(s, x): return x + s.b(x)
class Net2(nn.Module):
    def __init__(s):
        super().__init__(); ch = [64, 96, 192, 384, 512]; L = [nn.Conv2d(6, 64, 3, 1, 1)]
        for i in range(4): L += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), Res(ch[i + 1]), Res(ch[i + 1])]
        s.enc = nn.Sequential(*L, nn.GroupNorm(8, 512), nn.SiLU(), nn.Conv2d(512, 128, 1)); s.fc = nn.Sequential(nn.Flatten(), nn.Linear(8192, 3072), nn.SiLU(), nn.Linear(3072, 8192), nn.SiLU())
        D = [nn.Conv2d(128, 512, 3, 1, 1)]; dc = [512, 384, 192, 96]
        for i in range(3): D += [Res(dc[i]), nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dc[i], dc[i + 1], 3, 1, 1)]
        s.dec = nn.Sequential(*D, Res(96), nn.GroupNorm(8, 96), nn.SiLU(), nn.Conv2d(96, K * 3, 3, 1, 1))
    def forward(s, f): return torch.sigmoid(s.dec(s.fc(s.enc(f)).view(-1, 128, 8, 8))).view(-1, K, 3, 64, 64)
ck = f"../mode_probe/phase_net_{tag}.pt"
if not os.path.exists(ck):
    X = torch.cat([torch.load(f"../mode_probe/bank64k/part_{i}.pt") for i in range(NP)]); NTR = X.shape[0] - 512; Xtr, Xva = X[:NTR], X[NTR:]; print("bank", tuple(X.shape), flush=True)
    with torch.no_grad(): F0 = torch.cat([feat(Af(Xtr[i:i + 256].to(dev).float() / 255).abs()) for i in range(0, 2048, 256)]); fm, fs = F0.mean(0, keepdim=True), F0.std(0, keepdim=True) + 1e-3
    net = Net2().to(dev); print("params (M)", sum(p.numel() for p in net.parameters()) / 1e6, flush=True); BS = 128
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 6e-4, total_steps=EP * (NTR // BS), pct_start=0.1); t0 = time.time()
    def batch(xb):
        with torch.no_grad(): y = Af(xb).abs() + 0.05 * torch.randn(xb.shape[0], 3, 384, 384, device=dev); return (feat(y) - fm) / fs
    def loss_fn(p, t):
        e = ((p - t[:, None]) ** 2).mean(dim=(2, 3, 4)); return e.min(1).values.mean() + 0.05 * e.mean(), e
    for ep in range(EP):
        net.train(); perm = torch.randperm(NTR)
        for i in range(0, NTR - BS + 1, BS):
            xb = Xtr[perm[i:i + BS]].to(dev).float() / 255; fl = torch.rand(xb.shape[0], device=dev) < 0.5; xb = torch.where(fl.view(-1, 1, 1, 1), torch.flip(xb, dims=(3,)), xb); f = batch(xb)
            with torch.autocast("cuda", dtype=torch.bfloat16): p = net(f)
            l, _ = loss_fn(p.float(), low(xb)); opt.zero_grad(set_to_none=True); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            e = []; e32 = []
            for i in range(0, 512, 128):
                xv = Xva[i:i + 128].to(dev).float() / 255; p = net(batch(xv)); e.append(loss_fn(p, low(xv))[1]); e32.append(((torch.nn.functional.avg_pool2d(p.flatten(0, 1), 2).view(-1, K, 3, 32, 32) - torch.nn.functional.avg_pool2d(xv, 8)[:, None]) ** 2).mean(dim=(2, 3, 4)))
            e = torch.cat(e); e32 = torch.cat(e32)
        print(f"epoch {ep + 1}: train loss {float(l):.5f} | held-out: 64px PSNR best head {float(-10 * torch.log10(e.min(1).values).mean()):.2f}, mean head {float(-10 * torch.log10(e).mean()):.2f} | 32px best head {float(-10 * torch.log10(e32.min(1).values).mean()):.2f} | {time.time() - t0:.0f}s", flush=True)
        torch.save(dict(sd=net.state_dict(), fm=fm, fs=fs), ck + ".tmp")
    os.replace(ck + ".tmp", ck); del X, Xtr, Xva
C = torch.load(ck); net = Net2().to(dev); net.load_state_dict(C["sd"]); net.eval(); fm, fs = C["fm"], C["fs"]
# ---------------- test on real non-test images (same protocol and seeds as phase_net.py) ----------------
B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
R = torch.load("../mode_probe/reg_best.pt"); W, m_, s_ = R["W"].to(dev), R["m"].to(dev), R["s"].to(dev); OFF = torch.load("../mode_probe/face_bank_8192.pt")["rbar"].to(dev); rs = torch.load("../mode_probe/reg_resstd.pt").to(dev)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]; files = sorted(glob.glob(os.environ["VAL"] + "/**/*.png", recursive=True), key=nat)[:int(os.environ.get("NIMG", "100"))]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
ii = torch.arange(384, device=dev).float() - 192; RAD = (ii[:, None] ** 2 + ii[None, :] ** 2).sqrt()[None, None]
def fit(y, c0, M=None, bO=None):
    c = c0.clone()
    for k in (64, 128, 256, 512):
        ck_ = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck_], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); G = Af(mu + (ck_ @ Uf[:k]).view(1, 3, 256, 256)); l = (G.abs() + OFF - y) ** 2
            if M is not None: l = torch.where(M, (G - bO).abs() ** 2, l)
            l = l.mean() + 1e-4 * (ck_ ** 2 / var[:k]).mean(); l.backward(); return l
        opt.step(closure); c = torch.cat([ck_.detach(), c[k:]])
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
mloss = lambda xh, y: float(((Af(xh).abs() + OFF - y) ** 2).mean()); img = lambda c: (mu + (c @ Uf).view(1, 3, 256, 256)).clamp(0, 1)
BANDS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
def phacc(xh, Gt):
    Gh = Af(xh); cs = torch.cos(Gh.angle() - Gt.angle()); w = Gt.abs(); return [float((cs * w)[((RAD >= lo) & (RAD < hi)).expand_as(w)].sum() / w[((RAD >= lo) & (RAD < hi)).expand_as(w)].sum()) for lo, hi in BANDS]
out = []
for fi, f in enumerate(files):
    torch.manual_seed(5000 + fi); x = ld(f); Gt = Af(x); y = Gt.abs() + 0.05 * torch.randn(1, 3, 384, 384, device=dev); rec = {}
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; A = []
    for j in range(4):
        c = fit(y, creg + (0 if j == 0 else 1.0) * torch.randn_like(creg) * rs); xl = loop(img(c), y); A.append((mloss(img(c), y), ps(xl, x), phacc(xl, Gt)))
    with torch.no_grad(): H = up(net((feat(y) - fm) / fs)[0])
    rec["A"] = A; rec["net_psnr32"] = [ps(torch.nn.functional.avg_pool2d(H[k:k + 1], 8), torch.nn.functional.avg_pool2d(x, 8)) for k in range(K)]; rec["net_psnr"] = [ps(H[k:k + 1], x) for k in range(K)]; rec["net_acc"] = [phacc(H[k:k + 1], Gt) for k in range(K)]
    for r_ in (16, 32):
        M = (RAD < r_).expand(1, 3, 384, 384); Bc = []
        for k in range(K):
            bO = y * torch.exp(1j * Af(H[k:k + 1]).angle()); c = fit(y, creg, M, bO); c2 = fit(y, c); xl = loop(img(c2), y)
            xd = loop(H[k:k + 1], y, M, bO)                                                  # no eigenfaces: network image, denoiser loop with the band held softly
            Bc.append((mloss(img(c2), y), ps(xl, x), phacc(xl, Gt), mloss(xd, y), ps(xd, x)))
        rec[f"B{r_}"] = Bc
    out.append(rec)
json.dump(out, open(f"../mode_probe/logs_x/phnet_{tag}.json", "w"))
st = lambda v: f"mean {np.mean(v):5.2f} | 5th pct {np.percentile(v, 5):5.2f} | below 16 dB {(np.array(v) < 16).mean() * 100:3.0f}%"
pick = lambda Cc, j=0: Cc[int(np.argmin([c[j] for c in Cc]))]; A = [pick(r["A"]) for r in out]; cur = np.array([a[1] for a in A]); bad = cur < 17
print(f"=== {tag}: {len(out)} real non-test images ===\n   current 4 regression starts: {st(cur)}")
print(f"   network image: full-res PSNR best head {np.mean([max(r['net_psnr']) for r in out]):.2f}, mean head {np.mean([np.mean(r['net_psnr']) for r in out]):.2f} | 32px best head {np.mean([max(r['net_psnr32']) for r in out]):.2f}")
print("   phase accuracy [0-2, 2-4, 4-8, 8-16, 16-32, 32-64]: network " + " ".join(f"{v:.2f}" for v in np.mean([np.mean(r['net_acc'], 0) for r in out], 0)) + " | current coarse face " + " ".join(f"{v:.2f}" for v in np.mean([a[2] for a in A], 0)))
for r_ in (16, 32):
    b = np.array([pick(r[f"B{r_}"])[1] for r in out]); d = np.array([pick(r[f"B{r_}"], 3)[4] for r in out]); ob = np.array([max(c[1] for c in r[f"B{r_}"]) for r in out]); od = np.array([max(c[4] for c in r[f"B{r_}"]) for r in out])
    print(f"   phases r<{r_}: eigenface fit + refit + loop, pick by loss: {st(b)} | worst {b[bad].mean():.2f} (from {cur[bad].mean():.2f}) | best by truth {ob.mean():.2f}")
    print(f"   phases r<{r_}: network image + loop with band, pick by loss: {st(d)} | worst {d[bad].mean():.2f} | best by truth {od.mean():.2f}")
