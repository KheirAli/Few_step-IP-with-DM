import glob, re, math, numpy as np, torch
from PIL import Image
dev = torch.device("cuda:0"); nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
files = sorted(glob.glob("data100/**/*.png", recursive=True), key=nat)[:24]; mf = torch.load("mean_face01.pt").to(dev).float() * 2 - 1
Wp = lambda a: torch.remainder(a + 1, 2.0) - 1
def poisson(dx, dy, like):
    H, Wd = like.shape[-2:]; rho = torch.zeros_like(like); rho[..., :, :-1] += dx; rho[..., :, 1:] -= dx; rho[..., :-1, :] += dy; rho[..., 1:, :] -= dy
    ext = torch.cat([rho, rho.flip(-1)], -1); ext = torch.cat([ext, ext.flip(-2)], -2)
    ky = torch.arange(2 * H, device=like.device).float(); kx = torch.arange(2 * Wd, device=like.device).float()
    eig = (2 * torch.cos(math.pi * ky / H) - 2)[:, None] + (2 * torch.cos(math.pi * kx / Wd) - 2)[None, :]; eig[0, 0] = 1.0
    Fq = torch.fft.fft2(ext) / eig; Fq[..., 0, 0] = 0; return torch.fft.ifft2(Fq).real[..., :H, :Wd]
def unwrap(y, iters=0, const="range", smooth=0):
    ys = y
    if smooth > 0:                                              # wrap-aware smoothing of the measurement before taking differences (noise makes false wraps)
        import torchvision.transforms.functional as TF
        a = math.pi * y; ys = torch.atan2(TF.gaussian_blur(torch.sin(a), 5, smooth), TF.gaussian_blur(torch.cos(a), 5, smooth)) / math.pi
    phi = poisson(Wp(ys[..., :, 1:] - ys[..., :, :-1]), Wp(ys[..., 1:, :] - ys[..., :-1, :]), y)
    for _ in range(iters):                                      # re-unwrap what is left
        r = Wp(ys - phi); r = r - r.mean(dim=(2, 3), keepdim=True) * 0; phi = phi + poisson(Wp(r[..., :, 1:] - r[..., :, :-1]), Wp(r[..., 1:, :] - r[..., :-1, :]), y)
    ang = math.pi * (y - phi); c = torch.atan2(torch.sin(ang).mean(dim=(2, 3), keepdim=True), torch.cos(ang).mean(dim=(2, 3), keepdim=True)) / math.pi
    cand = torch.stack([y + 2 * torch.round((phi + c + 2 * j - y) / 2) for j in (-1, 0, 1)])          # three candidate images (2x scale), one per global offset
    if const == "range": sc = (cand.abs() - 2).clamp(min=0).mean(dim=(3, 4), keepdim=True)
    else: sc = ((cand / 2).clamp(-1, 1) - mf[None]).pow(2).mean(dim=(3, 4), keepdim=True) + 10 * (cand.abs() - 2).clamp(min=0).mean(dim=(3, 4), keepdim=True)
    return torch.gather(cand, 0, sc.argmin(0, keepdim=True).expand(1, *cand.shape[1:]))[0]
ps = lambda a, b: float(10 * torch.log10(4 / ((a - b) ** 2).mean()))
res = {}
for fi, f in enumerate(files):
    x = (torch.from_numpy(np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1).to(dev); torch.manual_seed(fi)
    y = Wp(2 * x) + 0.1 * torch.randn_like(x); ktrue = torch.round((2 * x - Wp(2 * x)) / 2)
    for name, kw in [("LS, range const", {}), ("LS, prior const", dict(const="prior")), ("LS x3, prior", dict(iters=3, const="prior")), ("smooth 1.0, prior", dict(smooth=1.0, const="prior")), ("smooth 1.0 x3, prior", dict(smooth=1.0, iters=3, const="prior")), ("smooth 2.0 x3, prior", dict(smooth=2.0, iters=3, const="prior"))]:
        u = unwrap(y, **kw); xh = (u / 2).clamp(-1, 1); kh = torch.round((u - y) / 2); res.setdefault(name, []).append((ps(xh, x), float((kh != torch.round((2 * x - (y - 0)) / 2)).float().mean())))
    res.setdefault("oracle branch (ceiling)", []).append((ps(((y + 2 * torch.round((2 * x - y) / 2)) / 2).clamp(-1, 1), x), 0.0))
for k, v in res.items(): v = np.array(v); print(f"UNW {k:26s} PSNR {v[:,0].mean():.2f} (worst {v[:,0].min():.1f}) | pixels with the wrong wrap count {v[:,1].mean()*100:.2f}%")

# ---- robust (L1-type) unwrapping: iteratively reweighted least squares, weighted Poisson solved by preconditioned CG ----
def grad(p): return p[..., :, 1:] - p[..., :, :-1], p[..., 1:, :] - p[..., :-1, :]
def div(gx, gy, like):
    r = torch.zeros_like(like); r[..., :, :-1] += gx; r[..., :, 1:] -= gx; r[..., :-1, :] += gy; r[..., 1:, :] -= gy; return r      # = -D'(g)
def pre(r):                                                     # unweighted Poisson solve as preconditioner (r = D'g form)
    H, Wd = r.shape[-2:]; ext = torch.cat([r, r.flip(-1)], -1); ext = torch.cat([ext, ext.flip(-2)], -2)
    ky = torch.arange(2 * H, device=r.device).float(); kx = torch.arange(2 * Wd, device=r.device).float()
    eig = (2 - 2 * torch.cos(math.pi * ky / H))[:, None] + (2 - 2 * torch.cos(math.pi * kx / Wd))[None, :]; eig[0, 0] = 1.0
    Fq = torch.fft.fft2(ext) / eig; Fq[..., 0, 0] = 0; return torch.fft.ifft2(Fq).real[..., :H, :Wd]
def unwrap_l1(y, outer=6, inner=40, eps=0.05, const="range", joint=False):
    dx, dy = Wp(y[..., :, 1:] - y[..., :, :-1]), Wp(y[..., 1:, :] - y[..., :-1, :]); phi = pre(-div(dx, dy, y)); wx = torch.ones_like(dx); wy = torch.ones_like(dy)
    for _ in range(outer):
        gx, gy = grad(phi); rx, ry = gx - dx, gy - dy
        if joint: rx = rx.abs().amax(1, keepdim=True).expand_as(rx); ry = ry.abs().amax(1, keepdim=True).expand_as(ry)   # an edge that breaks the rule in one colour is suspect in all
        wx = 1 / (rx.abs() + eps); wy = 1 / (ry.abs() + eps)
        Aop = lambda p: -div(wx * grad(p)[0], wy * grad(p)[1], p); b = -div(wx * dx, wy * dy, y)
        r = b - Aop(phi); z = pre(r); p = z.clone(); rz = (r * z).sum(dim=(2, 3), keepdim=True)
        for _ in range(inner):
            Ap = Aop(p); al = rz / (p * Ap).sum(dim=(2, 3), keepdim=True).clamp(min=1e-20); phi = phi + al * p; r = r - al * Ap; z = pre(r); rzn = (r * z).sum(dim=(2, 3), keepdim=True); p = z + (rzn / rz.clamp(min=1e-30)) * p; rz = rzn
    ang = math.pi * (y - phi); c = torch.atan2(torch.sin(ang).mean(dim=(2, 3), keepdim=True), torch.cos(ang).mean(dim=(2, 3), keepdim=True)) / math.pi
    cand = torch.stack([y + 2 * torch.round((phi + c + 2 * j - y) / 2) for j in (-1, 0, 1)])
    sc = (cand.abs() - 2).clamp(min=0).mean(dim=(3, 4), keepdim=True) if const == "range" else ((cand / 2).clamp(-1, 1) - mf[None]).pow(2).mean(dim=(3, 4), keepdim=True) + 10 * (cand.abs() - 2).clamp(min=0).mean(dim=(3, 4), keepdim=True)
    return torch.gather(cand, 0, sc.argmin(0, keepdim=True).expand(1, *cand.shape[1:]))[0]
res = {}
for fi, f in enumerate(files):
    x = (torch.from_numpy(np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1).to(dev); torch.manual_seed(fi)
    y = Wp(2 * x) + 0.1 * torch.randn_like(x); kt = torch.round((2 * x - y) / 2)
    viol = float(((grad(2 * x)[0].abs() > 1).float().mean() + (grad(2 * x)[1].abs() > 1).float().mean()) / 2)
    for name, kw in [("robust, range", {}), ("robust, prior const", dict(const="prior")), ("robust, eps 0.02, prior", dict(eps=0.02, const="prior")), ("robust joint colours, prior", dict(joint=True, const="prior")), ("robust joint, 10 outer", dict(joint=True, const="prior", outer=10))]:
        u = unwrap_l1(y, **kw); res.setdefault(name, []).append((ps((u / 2).clamp(-1, 1), x), float((torch.round((u - y) / 2) != kt).float().mean()), viol))
for k, v in res.items(): v = np.array(v); print(f"UNW2 {k:28s} PSNR {v[:,0].mean():.2f} (worst {v[:,0].min():.1f}, images above 30 dB {np.mean(v[:,0]>30)*100:.0f}%) | wrong wrap count {v[:,1].mean()*100:.2f}% | neighbouring pairs that break the half-period rule {v[:,2].mean()*100:.2f}%")
