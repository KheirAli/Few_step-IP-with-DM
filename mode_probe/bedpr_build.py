"""Bedroom versions of the files the unmasked phase-retrieval start stage reads (run from PnP-CM/).  MODEL-GENERATED images only:
every image is a 2-call sample of the bedroom consistency model, drawn exactly as pca_pr.py / reg_pr.py / local_pca.py draw faces.
  pca_cm_bed.pt          mean + 512 eigen-images of 2048 samples (seed 0)                                  [as pca_pr.py]
  reg_best_bed.pt        ridge map low-frequency log-magnitudes -> 512 coefficients, 8192 samples (seed 777);
                         lambda chosen on 512 HELD-OUT MODEL SAMPLES (as reg_pr_gen.py; unified.py only supports this feature set)
  reg_resstd_bed.pt      RMS held-out residual of that map per coefficient (this reproduces the face file reg_resstd.pt exactly)
  face_bank_8192_bed.pt  key rbar: average magnitude mismatch of the 512-component model over 8192 samples (seed 2024) [as local_pca.py];
                         the images themselves are not stored (the sampler reads only rbar)
No test image is read anywhere."""
import sys, os, yaml, numpy as np, torch
from argparse import Namespace
sys.path.insert(0, "."); import functions.fft_util as FU
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition, append_dims
dev = torch.device("cuda:0"); L = "../mode_probe"; BS = int(os.environ.get("BS", "16"))
def ns(d):
    n = Namespace()
    for k, v in d.items(): setattr(n, k, ns(v) if isinstance(v, dict) else v)
    return n
config = ns(yaml.safe_load(open("configs/lsun_bedroom_256.yml"))); md = dict(vars(config.model)); md["image_size"] = config.data.image_size
model = create_model(**md); model.load_state_dict(torch.load("exp/logs/lsun_bedroom/cd_bedroom256_lpips.pt", map_location=dev)); model.to(dev).eval()
def cm(z, Lv):
    Lv = torch.tensor([Lv], device=dev); cs, co, ci = [append_dims(s, z.ndim) for s in get_scalings_for_boundary_condition(Lv)]
    return torch.clamp(co * model(ci * z, (1000 * 0.25 * torch.log(Lv + 1e-44)).expand(z.shape[0])) + cs * z, -1, 1)
def draw():
    x = cm(80.0 * torch.randn(BS, 3, 256, 256, device=dev), 80.0); return (cm(x + 0.8 * torch.randn_like(x), 0.8) + 1) / 2
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); A = lambda im: FU.fft2_m(pad(im)).abs(); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
def save(obj, f): torch.save(obj, f + ".tmp"); os.replace(f + ".tmp", f)
# ---- eigen-images
out = f"{L}/pca_cm_bed.pt"
if not os.path.exists(out):
    torch.manual_seed(0); X = []
    with torch.no_grad():
        for _ in range(2048 // BS): X.append(draw().cpu())
    X = torch.cat(X).flatten(1); mu = X.mean(0); Xc = (X - mu).to(dev)
    ev, V = torch.linalg.eigh(Xc @ Xc.T); idx = torch.argsort(ev, descending=True)[:512]; U = (Xc.T @ V[:, idx]) / ev[idx].clamp(min=1e-8).sqrt()
    save({"mu": mu.view(1, 3, 256, 256), "U": U.T.reshape(-1, 3, 256, 256).cpu(), "var": (ev[idx] / X.shape[0]).cpu()}, out); del X, Xc, V, U; print("BUILD pca done", flush=True)
B_ = torch.load(out); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
# ---- regression start
N = 8192; bank = f"{L}/reg_bank_{N}_bed.pt"
if not os.path.exists(bank):
    torch.manual_seed(777); F1, C, R = [], [], []
    with torch.no_grad():
        for _ in range((N + 512) // BS):
            x = draw(); y = A(x); y = y + 0.05 * torch.randn_like(y); lm = torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1)
            c = (x - mu).flatten(1) @ Uf.T; F1.append(lm.cpu()); C.append(c.cpu()); R.append(torch.stack([10 * torch.log10(1 / ((x[i] - (mu[0] + (c[i] @ Uf).view(3, 256, 256))) ** 2).mean()) for i in range(BS)]).cpu())
    save({"lm": torch.cat(F1), "c": torch.cat(C), "ceil": torch.cat(R)}, bank); print("BUILD reg bank done", flush=True)
D = torch.load(bank); lm, C = D["lm"].to(dev), D["c"].to(dev); img = lambda c: (mu + (c @ Uf).view(-1, 3, 256, 256)).clamp(0, 1)
print(f"BUILD 512-component ceiling on model samples {float(D['ceil'].mean()):.2f} dB | mean image vs held-out samples' projections {np.mean([ps(mu, img(C[N + i:N + i + 1])) for i in range(512)]):.2f} dB", flush=True)
m_, s_ = lm[:N].mean(0), lm[:N].std(0).clamp(min=1e-6); X = (lm - m_) / s_; Xtr, Xva = X[:N], X[N:]; K = Xtr @ Xtr.T; best = (-1, None)
for lam in (1e2, 1e3, 1e4, 1e5):
    W = Xtr.T @ torch.linalg.solve(K + lam * torch.eye(N, device=dev), C[:N]); pv = Xva @ W; v = float(np.mean([ps(img(pv[i:i + 1]), img(C[N + i:N + i + 1])) for i in range(512)]))
    print(f"BUILD ridge lambda {lam:g}: held-out model samples (vs their projection) {v:.2f} dB", flush=True)
    if v > best[0]: best = (v, {"W": W.cpu(), "m": m_.cpu(), "s": s_.cpu(), "feat": "low-frequency log-magnitudes", "kind": "ridge", "lam": lam}, ((C[N:] - pv) ** 2).mean(0).sqrt().cpu())
print(f"BUILD chosen lambda {best[1]['lam']:g} ({best[0]:.2f} dB)", flush=True); save(best[1], f"{L}/reg_best_bed.pt"); save(best[2], f"{L}/reg_resstd_bed.pt")
# ---- average mismatch of the eigen-image magnitude model
out = f"{L}/face_bank_8192_bed.pt"
if not os.path.exists(out):
    torch.manual_seed(2024); rb = 0
    with torch.no_grad():
        for _ in range(8192 // BS):
            x = draw(); a = A(x); c = (x - mu).flatten(1) @ Uf.T; rb = rb + (a - A(mu + (c @ Uf).view(-1, 3, 256, 256))).sum(0, keepdim=True)
    save({"rbar": (rb / 8192).cpu()}, out)
print("BUILD ALLDONE", flush=True)
