"""Eigenface route for UNMASKED (DPS/DAPS) phase retrieval.  Run from the PnP-CM repo root.
1. Build a face subspace from 2048 samples of the consistency model itself (no real images, so no leakage of test data).
2. Oracle: how well can k components represent a test face at all?
3. Fit: recover the k coefficients from the Fourier magnitudes alone (L-BFGS on the true magnitude loss, k grown 8 -> 256,
   each stage started from the previous one; start = mean face).  No ground truth is used in the fit."""
import sys, os, glob, re, yaml, numpy as np, torch
from argparse import Namespace
from PIL import Image
sys.path.insert(0, ".")
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition, append_dims
import functions.fft_util as FU
dev = torch.device("cuda:0"); out = "../mode_probe/pca_cm.pt"
if not os.path.exists(out):
    def ns(d):
        n = Namespace()
        for k, v in d.items(): setattr(n, k, ns(v) if isinstance(v, dict) else v)
        return n
    config = ns(yaml.safe_load(open("configs/celeba_hq_256.yml"))); md = dict(vars(config.model)); md["image_size"] = config.data.image_size
    model = create_model(**md); model.load_state_dict(torch.load("exp/logs/celeba_hq/ema_0.9999432189950708_1175000.pt", map_location=dev)); model.to(dev).eval()
    def cm(z, L):
        L = torch.tensor([L], device=dev); cs, co, ci = [append_dims(s, z.ndim) for s in get_scalings_for_boundary_condition(L)]
        return torch.clamp(co * model(ci * z, (1000 * 0.25 * torch.log(L + 1e-44)).expand(z.shape[0])) + cs * z, -1, 1)
    torch.manual_seed(0); X = []
    with torch.no_grad():
        for _ in range(128):
            x = cm(80.0 * torch.randn(16, 3, 256, 256, device=dev), 80.0); x = cm(x + 0.8 * torch.randn_like(x), 0.8)     # two-step samples
            X.append(((x + 1) / 2).cpu())
    X = torch.cat(X).flatten(1); mu = X.mean(0); Xc = (X - mu).to(dev)
    ev, V = torch.linalg.eigh(Xc @ Xc.T); idx = torch.argsort(ev, descending=True)[:512]
    U = (Xc.T @ V[:, idx]) / ev[idx].clamp(min=1e-8).sqrt()
    torch.save({"mu": mu.view(1, 3, 256, 256), "U": U.T.reshape(-1, 3, 256, 256).cpu(), "var": (ev[idx] / X.shape[0]).cpu()}, out); del model
B = torch.load(out); mu, U, var = B["mu"].to(dev), B["U"].to(dev), B["var"].to(dev)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
files = sorted(glob.glob("../mode_probe/data_paper/celeba_hq/celeba_hq/*.png"), key=nat)[:int(os.environ.get("NIMG", "12"))]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
KS = [8, 16, 32, 64, 128, 256, 512]; R = {k: [] for k in KS}; O = {k: [] for k in KS}; M = {k: [] for k in KS}; FL = {k: [] for k in KS}; res = {}
torch.manual_seed(1)
for fi, f in enumerate(files):
    x = ld(f); amp = FU.fft2_m(pad(x)).abs(); y = amp + 0.05 * torch.randn_like(amp)
    c = torch.zeros(0, device=dev)
    for k in KS:
        Uk = U[:k]; O[k].append(ps((mu + (((x - mu).flatten() @ Uk.flatten(1).T) @ Uk.flatten(1)).view(1, 3, 256, 256)).clamp(0, 1), x))
        c = torch.cat([c, torch.zeros(k - c.numel(), device=dev)]).detach().requires_grad_(True)
        opt = torch.optim.LBFGS([c], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); img = mu + (c @ Uk.flatten(1)).view(1, 3, 256, 256)
            loss = ((FU.fft2_m(pad(img)).abs() - y) ** 2).mean() + 1e-4 * (c ** 2 / var[:k]).mean(); loss.backward(); return loss
        opt.step(closure)
        img = (mu + (c.detach() @ Uk.flatten(1)).view(1, 3, 256, 256)).clamp(0, 1)
        R[k].append(ps(img, x)); FL[k].append(ps(torch.flip(img, dims=(2, 3)), x)); M[k].append(float((FU.fft2_m(pad(img)).abs() - y).norm() / y.norm()))
    res[os.path.basename(f)] = c.detach().cpu()
torch.save(res, "../mode_probe/pca_fit_coeffs.pt")
print(f"{len(files)} images | mean face alone: {np.mean([ps(mu, ld(f)) for f in files]):.2f} dB")
for k in KS:
    print(f"  k={k:3d}: best possible with k components {np.mean(O[k]):.2f} dB | fitted from magnitudes only {np.mean(R[k]):.2f} dB (worst {np.min(R[k]):.1f}, best {np.max(R[k]):.1f}; flipped would be {np.mean(FL[k]):.2f}) | misfit {np.mean(M[k]):.3f}")
