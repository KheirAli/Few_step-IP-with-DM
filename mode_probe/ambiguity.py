"""AMBIGUITY CHECK for a coordinate-wise measurement function f (run from PnP-CM/).  For each coordinate, given the measurement y and the current
estimate v, the data-fit objective (1/2)(f(c)-y)^2 + (rho/2)(c-v)^2 is evaluated on a grid.  Values within a factor exp(-4) of the best likelihood
are 'plausible'.  If the plausible set comes in 2+ separate pieces for many coordinates, the measurement does not single out one answer and a
search over starting points is needed; a single piece means the data-fit plus the prior should be enough."""
import glob, numpy as np, torch
from PIL import Image
dev = torch.device("cuda:0"); fs = sorted(glob.glob("../mode_probe/data_gen512/celeba_hq/celeba_hq/*.png"))[:3]; K = 257; sy = 0.1
X = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) for f in fs]).to(dev) * 2 - 1
g = torch.Generator(device=dev).manual_seed(0); tau = torch.rand(X.shape, device=dev, generator=g) * 2 - 1
OPS = {"HDR (clip)": (lambda z: (2 * z).clamp(-1, 1), None), "quantisation, 8 levels": (lambda z: (torch.floor((z.clamp(-1, 1 - 1e-6) + 1) * 4) + 0.5) / 4 - 1, None),
       "one-bit": (lambda z: torch.sign(z - tau), None), "tone curve": (lambda z: 2 * ((z.clamp(-1, 1) + 1) / 2 + 1e-4) ** 0.4 - 1, None),
       "exponential attenuation (CT ray)": (lambda z: 10 * (1 - torch.exp(-z)), (0.0, 8.0)), "wrap-around": (lambda z: torch.remainder(2 * z + 1, 2) - 1, None),
       "magnitude (real coefficient, sign lost)": (lambda z: z.abs(), None)}
for name, (f, rng) in OPS.items():
    out = []
    for s_v in (0.5, 0.15):
        lo_, hi_ = rng or (-1.0, 1.0); c = torch.linspace(lo_, hi_, K, device=dev).view(K, 1, 1, 1, 1); xx = (X + 1) * 2 if rng else X; sc = s_v * (hi_ - lo_) / 2
        y = f(xx) + sy * torch.randn(xx.shape, device=dev, generator=g); v = xx + sc * torch.randn(xx.shape, device=dev, generator=g); rho = sy ** 2 / sc ** 2
        ob = 0.5 * (f(c.expand(K, *xx.shape)) - y[None]) ** 2 + 0.5 * rho * (c - v[None]) ** 2; good = ob <= ob.min(0, keepdim=True)[0] + 4 * sy ** 2
        pieces = (good[1:] & ~good[:-1]).sum(0) + good[0].long(); zs = c.expand(K, *xx.shape).gather(0, ob.argmin(0, keepdim=True))[0]
        out.append(f"{100 * float((pieces >= 2).float().mean()):5.1f}% of coordinates ambiguous, error of the best value {float(((zs - xx) ** 2).mean().sqrt() / ((hi_ - lo_) / 2)):.3f} (estimate off by {s_v})")
    print(f"AMB {name:40s} " + " | ".join(out), flush=True)
