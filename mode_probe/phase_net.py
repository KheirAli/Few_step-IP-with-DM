"""Gate 2/3 for 'phase retrieval -> partial Fourier reconstruction'.  A small network, trained ONLY on samples of the consistency model,
maps noisy Fourier magnitudes to K low-resolution face hypotheses (32x32); their low-frequency phases are then used as soft linear
constraints in the eigenface fit.  Tested on real non-test images."""
import sys, os, glob, re, json, time, numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); K = int(os.environ.get("KH", "4")); EP = int(os.environ.get("EP", "150")); BIG = os.environ.get("BIG", "0") == "1"; tag = f"k{K}" + ("big" if BIG else "") + os.environ.get("SUF", ""); NTR = 63488 if BIG else 7680
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); crop = lambda g: g[..., 64:320, 64:320]; Af = lambda im: FU.fft2_m(pad(im)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
feat = lambda y: torch.log1p(y.clamp(min=0))[..., 128:256, 128:256]; low = lambda x: torch.nn.functional.avg_pool2d(x, 8); up = lambda z: torch.nn.functional.interpolate(z, size=256, mode="bicubic", align_corners=False).clamp(0, 1)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); ch = [3, 64, 128, 256, 256, 256]; L = []
        for i in range(5): L += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), nn.GroupNorm(8, ch[i + 1]), nn.SiLU()]
        s.enc = nn.Sequential(*L); s.fc = nn.Sequential(nn.Flatten(), nn.Linear(4096, 2048), nn.SiLU(), nn.Linear(2048, K * 3 * 32 * 32))
    def forward(s, f): return torch.sigmoid(s.fc(s.enc(f))).view(-1, K, 3, 32, 32)
ck = f"../mode_probe/phase_net_{tag}.pt"
if not os.path.exists(ck):
    X = torch.cat([torch.load(f"../mode_probe/bank64k/part_{i}.pt") for i in range(8)]) if BIG else torch.load("../mode_probe/face_bank_8192.pt")["x"]
    SC = 255.0 if BIG else 1.0; Xtr, Xva = X[:NTR], X[NTR:NTR + 512]; print("bank", tuple(X.shape), flush=True)
    with torch.no_grad(): F0 = torch.cat([feat(Af(Xtr[i:i + 256].to(dev).float() / SC).abs()) for i in range(0, 2048, 256)]); fm, fs = F0.mean(0, keepdim=True), F0.std(0, keepdim=True) + 1e-3
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * (NTR // 64)); t0 = time.time()
    def batch(xb):
        y = Af(xb).abs() + 0.05 * torch.randn(xb.shape[0], 3, 384, 384, device=dev); return (feat(y) - fm) / fs
    def loss_fn(p, t):
        e = ((p - t[:, None]) ** 2).mean(dim=(2, 3, 4)); return e.min(1).values.mean() + 0.05 * e.mean(), e
    for ep in range(EP):
        net.train(); perm = torch.randperm(NTR)
        for i in range(0, NTR, 64):
            xb = Xtr[perm[i:i + 64]].to(dev).float() / SC; fl = torch.rand(xb.shape[0], device=dev) < 0.5; xb = torch.where(fl.view(-1, 1, 1, 1), torch.flip(xb, dims=(3,)), xb)
            l, _ = loss_fn(net(batch(xb)), low(xb)); opt.zero_grad(); l.backward(); opt.step(); sched.step()
        if ep % max(EP // 6, 1) == max(EP // 6, 1) - 1 or ep == EP - 1:
            net.eval()
            with torch.no_grad(): e = torch.cat([loss_fn(net(batch(Xva[i:i + 128].to(dev).float() / SC)), low(Xva[i:i + 128].to(dev).float() / SC))[1] for i in range(0, 512, 128)])
            print(f"epoch {ep + 1}: train loss {float(l):.5f} | held-out samples: low-res PSNR best head {float(-10 * torch.log10(e.min(1).values).mean()):.2f}, mean head {float(-10 * torch.log10(e).mean()):.2f} | {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(sd=net.state_dict(), fm=fm, fs=fs), ck)
C = torch.load(ck); net = Net().to(dev); net.load_state_dict(C["sd"]); net.eval(); fm, fs = C["fm"], C["fs"]
# ---------------- test on real non-test images ----------------
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
def soft(x, y, r=1.0):
    G = Af(x); return crop(FU.ifft2_m(G / (G.abs() + 1e-8) * (y + r * G.abs()) / (1 + r)).real).clamp(0, 1)
def tv(x, lam=0.05, n=30):
    px = torch.zeros_like(x); py = torch.zeros_like(x)
    for _ in range(n):
        u = x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2)); gx = torch.roll(u, -1, 3) - u; gy = torch.roll(u, -1, 2) - u; nrm = 1 + 0.25 / lam * (gx ** 2 + gy ** 2).sqrt()
        px = (px + 0.25 / lam * gx) / nrm; py = (py + 0.25 / lam * gy) / nrm
    return (x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2))).clamp(0, 1)
def loop(x0, y):
    xx = x0.clone()
    for _ in range(40): xx = tv(soft(xx, y))
    return xx
mloss = lambda xh, y: float(((Af(xh).abs() + OFF - y) ** 2).mean()); img = lambda c: (mu + (c @ Uf).view(1, 3, 256, 256)).clamp(0, 1)
BANDS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32)]
def phacc(xh, Gt):
    Gh = Af(xh); cs = torch.cos(Gh.angle() - Gt.angle()); w = Gt.abs(); return [float((cs * w)[((RAD >= lo) & (RAD < hi)).expand_as(w)].sum() / w[((RAD >= lo) & (RAD < hi)).expand_as(w)].sum()) for lo, hi in BANDS]
out = []; t_net = 0
for fi, f in enumerate(files):
    torch.manual_seed(5000 + fi); x = ld(f); Gt = Af(x); y = Gt.abs() + 0.05 * torch.randn(1, 3, 384, 384, device=dev); rec = {}
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; A = []
    for j in range(4):
        c = fit(y, creg + (0 if j == 0 else 1.0) * torch.randn_like(creg) * rs); xl = loop(img(c), y); A.append((mloss(img(c), y), ps(xl, x), phacc(xl, Gt)))
    with torch.no_grad(): t = time.time(); H = up(net((feat(y) - fm) / fs)[0]); torch.cuda.synchronize(); t_net += time.time() - t
    rec["A"] = A; rec["net_lowres_psnr"] = [ps(low(H[k:k + 1]), low(x)) for k in range(K)]; rec["net_acc"] = [phacc(H[k:k + 1], Gt) for k in range(K)]
    for r_ in (8, 16):
        M = (RAD < r_).expand(1, 3, 384, 384); Bc = []
        for k in range(K):
            bO = y * torch.exp(1j * Af(H[k:k + 1]).angle()); c = fit(y, creg, M, bO); c2 = fit(y, c)       # band-constrained fit, then free refit (phases not locked)
            xl = loop(img(c2), y); Bc.append((mloss(img(c2), y), ps(xl, x), phacc(xl, Gt), ps(img(c), x)))
        rec[f"B{r_}"] = Bc
    out.append(rec)
json.dump(out, open(f"../mode_probe/logs_x/phnet_{tag}.json", "w")); print(f"network time per image {t_net / len(files) * 1000:.1f} ms")
