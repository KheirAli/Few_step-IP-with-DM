"""Offline gate for the learned model-mismatch correction (unmasked phase retrieval).  Run from PnP-CM root.
r = |F(x)| - |F(mu + U c)| is the part of the magnitudes the eigenface model cannot produce.  Can r be PREDICTED from c?
Ridge regression of r on c (linear, and with products of the 32 leading coefficients), trained on 4096 fresh model samples,
judged on 512 held-out ones by the share of the variance of r explained (0 = no better than its average, 1 = perfect)."""
import sys, yaml, numpy as np, torch
from argparse import Namespace
sys.path.insert(0, "."); import functions.fft_util as FU
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition, append_dims
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U = B_["mu"].to(dev), B_["U"].to(dev); Uf = U.flatten(1)
def ns(d):
    n = Namespace()
    for k, v in d.items(): setattr(n, k, ns(v) if isinstance(v, dict) else v)
    return n
config = ns(yaml.safe_load(open("configs/celeba_hq_256.yml"))); md = dict(vars(config.model)); md["image_size"] = config.data.image_size
model = create_model(**md); model.load_state_dict(torch.load("exp/logs/celeba_hq/ema_0.9999432189950708_1175000.pt", map_location=dev)); model.to(dev).eval()
def cm(z, L):
    L = torch.tensor([L], device=dev); cs, co, ci = [append_dims(s, z.ndim) for s in get_scalings_for_boundary_condition(L)]
    return torch.clamp(co * model(ci * z, (1000 * 0.25 * torch.log(L + 1e-44)).expand(z.shape[0])) + cs * z, -1, 1)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); A = lambda im: FU.fft2_m(pad(im)).abs()
iu = torch.triu_indices(32, 32)
def feat(c): return torch.cat([torch.ones(c.shape[0], 1, device=dev), c, (c[:, :32, None] * c[:, None, :32])[:, iu[0], iu[1]]], 1)
def batch():
    with torch.no_grad():
        x = cm(80.0 * torch.randn(16, 3, 256, 256, device=dev), 80.0); x = (cm(x + 0.8 * torch.randn_like(x), 0.8) + 1) / 2
        c = (x - mu).flatten(1) @ Uf.T; xp = mu + (c @ Uf).view(-1, 3, 256, 256); a = A(x); return c, (a - A(xp)).flatten(1), a.flatten(1)
torch.manual_seed(999); Sxx = 0; Sxr = 0; n = 0
for _ in range(256):
    c, r, a = batch(); X = feat(c).double(); Sxx = Sxx + X.T @ X; Sxr = Sxr + (X.T @ r.double()).float(); n += 16
VAL = [batch() for _ in range(32)]; cv = torch.cat([v[0] for v in VAL]); rv = torch.cat([v[1] for v in VAL]); av = torch.cat([v[2] for v in VAL]); Xv = feat(cv)
rbar = Sxr[0] / n
f = torch.fft.fftshift(torch.fft.fftfreq(384, device=dev)); rad = (f[:, None] ** 2 + f[None, :] ** 2).sqrt().flatten().repeat(3)
print(f"trained on {n} samples, judged on {rv.shape[0]} held-out samples")
print(f"size of the mismatch: rms {float(rv.pow(2).mean().sqrt()):.4f} against rms magnitude {float(av.pow(2).mean().sqrt()):.4f}; measurement noise is 0.05")
tot = ((rv - rbar) ** 2).sum()
for name, cols in [("linear in the 512 coefficients", slice(0, 513)), ("plus products of the 32 leading coefficients", slice(0, Xv.shape[1]))]:
    for lam in (1e-1, 1e1, 1e3):
        k = cols.stop; Wm = torch.linalg.solve(Sxx[:k, :k] + lam * torch.eye(k, device=dev, dtype=torch.float64), Sxr[:k].double()).float(); pr = Xv[:, :k] @ Wm
        ev = 1 - float(((rv - pr) ** 2).sum() / tot)
        bands = "  ".join(f"{lo:.2f}-{hi:.2f}: {1 - float(((rv - pr)[:, (rad >= lo) & (rad < hi)] ** 2).sum() / ((rv - rbar)[:, (rad >= lo) & (rad < hi)] ** 2).sum()):+.3f}" for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.8)])
        print(f"   {name:46s} lambda {lam:g}: variance explained {ev:+.3f} | by frequency band  {bands}")
print(f"share of the mismatch that is its plain average (same for every face): {float((rbar ** 2).sum() * rv.shape[0] / (rv ** 2).sum()):.3f}")
