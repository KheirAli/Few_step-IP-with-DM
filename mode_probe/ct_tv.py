"""Classical baselines for the sparse-view CT test (run from PnP-CM root): regularised least squares, and total-variation regularised
reconstruction (alternating the exact CG data-fit and a TV denoiser).  Settings chosen on 24 non-test images, then the 107 test images."""
import sys, glob, re, math, itertools, numpy as np, torch, lpips
from PIL import Image
dev = torch.device("cuda:0"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev); nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
def radon(x, K):
    xp = torch.nn.functional.pad(x, (54, 54, 54, 54)).expand(K, -1, -1, -1); th = torch.arange(K, device=x.device).float() * math.pi / K
    M = torch.stack([torch.stack([th.cos(), -th.sin(), torch.zeros_like(th)], 1), torch.stack([th.sin(), th.cos(), torch.zeros_like(th)], 1)], 1)
    rot = torch.nn.functional.grid_sample(xp, torch.nn.functional.affine_grid(M, list(xp.shape), align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    return rot.sum(dim=2).permute(1, 0, 2)[None] / 16.0
def At(r, like, K):
    with torch.enable_grad():
        z = torch.zeros_like(like).requires_grad_(True); return torch.autograd.grad((radon(z, K) * r).sum(), z)[0]
def prox(y, v, rho, K, n=12):
    b = At(y, v, K) + rho * v; op = lambda z: At(radon(z, K), v, K) + rho * z; z = v.clone(); r = b - op(z); p = r.clone(); rs = (r * r).sum()
    for _ in range(n):
        Ap = op(p); al = rs / (p * Ap).sum().clamp(min=1e-20); z = z + al * p; r = r - al * Ap; rn = (r * r).sum(); p = r + (rn / rs) * p; rs = rn
    return z
def tv(x, lam, n=30):
    px = torch.zeros_like(x); py = torch.zeros_like(x)
    for _ in range(n):
        u = x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2)); gx = torch.roll(u, -1, 3) - u; gy = torch.roll(u, -1, 2) - u; nrm = 1 + 0.25 / lam * (gx ** 2 + gy ** 2).sqrt()
        px = (px + 0.25 / lam * gx) / nrm; py = (py + 0.25 / lam * gy) / nrm
    return x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2))
ld = lambda p: (torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1).to(dev)
def tvrec(y, K, rho, lam, its=30):
    v = torch.zeros(1, 3, 256, 256, device=dev)
    for _ in range(its): v = tv(prox(y, v, rho, K), lam)
    return v.clamp(-1, 1)
ps = lambda a, b: float(10 * torch.log10(4 / ((a - b) ** 2).mean()))
sel = sorted(glob.glob("../mode_probe/data100/**/*.png", recursive=True), key=nat)[:24]; test = sorted(glob.glob("../mode_probe/data_paper/**/*.png", recursive=True), key=nat)[:107]
for K in (60, 30):
    torch.manual_seed(1); Y = [(ld(f), None) for f in sel[:12]]; Y = [(x, radon(x, K) + 0.1 * torch.randn_like(radon(x, K))) for x, _ in Y]
    grid = {(rho, lam): np.mean([ps(tvrec(y, K, rho, lam, 15), x) for x, y in Y]) for rho, lam in itertools.product((0.3, 1.0, 3.0), (0.01, 0.02, 0.04, 0.08))}; best = max(grid, key=grid.get)
    rr = {r0: np.mean([ps(prox(y, torch.zeros_like(x), r0, K, 40).clamp(-1, 1), x) for x, y in Y]) for r0 in (0.02, 0.05, 0.1, 0.3)}; rb = max(rr, key=rr.get)
    torch.manual_seed(2); R = []; T = []
    for f in test:
        x = ld(f); y = radon(x, K) + 0.1 * torch.randn_like(radon(x, K)); a = prox(y, torch.zeros_like(x), rb, K, 40).clamp(-1, 1); b = tvrec(y, K, *best)
        with torch.no_grad(): R.append((ps(a, x), float(lp(a, x)))); T.append((ps(b, x), float(lp(b, x))))
    R = np.array(R); T = np.array(T)
    print(f"CLASSICAL {K} views: regularised least squares (penalty {rb}) {R[:,0].mean():.2f}/{R[:,1].mean():.3f} | total variation (penalty {best[0]}, strength {best[1]}) {T[:,0].mean():.2f}/{T[:,1].mean():.3f} n={len(T)}", flush=True)
