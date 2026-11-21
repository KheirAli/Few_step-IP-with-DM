"""Calibrated inverse map for UNMASKED phase retrieval (run from PnP-CM root).
Training pairs come only from samples of the consistency model: (features of the noisy Fourier magnitudes of the COMPLETE face)
-> (its 512 eigenface coefficients).  Ridge and kernel ridge regression; tested on held-out samples and on 12 real test faces."""
import sys, os, glob, re, yaml, numpy as np, torch
from argparse import Namespace
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U = B_["mu"].to(dev), B_["U"].to(dev); Uf = U.flatten(1)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
def feats(y):                                                  # y: [B,3,384,384] noisy magnitudes
    lm = torch.log1p(y.clamp(min=0))[..., 192 - 32:192 + 32, 192 - 32:192 + 32].flatten(1)            # low-frequency log-magnitudes
    a = FU.ifft2_m((y.clamp(min=0) ** 2).to(torch.complex64)).real; a = a / a[..., 192:193, 192:193].clamp(min=1e-8)
    ac = a[..., 192 - 24:192 + 25, 192 - 24:192 + 25].flatten(1)                                     # short-lag autocorrelation
    return lm, ac
N = int(os.environ.get("NTRAIN", "8192")); bank = f"../mode_probe/reg_bank_{N}.pt"
if not os.path.exists(bank):
    from guided_diffusion.script_util import create_model
    from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition, append_dims
    def ns(d):
        n = Namespace()
        for k, v in d.items(): setattr(n, k, ns(v) if isinstance(v, dict) else v)
        return n
    config = ns(yaml.safe_load(open("configs/celeba_hq_256.yml"))); md = dict(vars(config.model)); md["image_size"] = config.data.image_size
    model = create_model(**md); model.load_state_dict(torch.load("exp/logs/celeba_hq/ema_0.9999432189950708_1175000.pt", map_location=dev)); model.to(dev).eval()
    def cm(z, L):
        L = torch.tensor([L], device=dev); cs, co, ci = [append_dims(s, z.ndim) for s in get_scalings_for_boundary_condition(L)]
        return torch.clamp(co * model(ci * z, (1000 * 0.25 * torch.log(L + 1e-44)).expand(z.shape[0])) + cs * z, -1, 1)
    torch.manual_seed(777); F1, F2, C, R = [], [], [], []
    with torch.no_grad():
        for _ in range((N + 512) // 16):
            x = cm(80.0 * torch.randn(16, 3, 256, 256, device=dev), 80.0); x = (cm(x + 0.8 * torch.randn_like(x), 0.8) + 1) / 2
            y = FU.fft2_m(pad(x)).abs(); y = y + 0.05 * torch.randn_like(y); lm, ac = feats(y)
            c = (x - mu).flatten(1) @ Uf.T; F1.append(lm.cpu()); F2.append(ac.cpu()); C.append(c.cpu())
            R.append(torch.stack([10 * torch.log10(1 / ((x[i] - (mu[0] + (c[i] @ Uf).view(3, 256, 256))) ** 2).mean()) for i in range(16)]).cpu())
    torch.save({"lm": torch.cat(F1), "ac": torch.cat(F2), "c": torch.cat(C), "ceil": torch.cat(R)}, bank); del model
D = torch.load(bank); lm, ac, C = D["lm"].to(dev), D["ac"].to(dev), D["c"].to(dev); ntr = N
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
files = sorted(glob.glob("../mode_probe/data_paper/celeba_hq/celeba_hq/*.png"), key=nat)[:int(os.environ.get("NIMG", "12"))]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
torch.manual_seed(1); T = []
for f in files:
    x = ld(f); y = FU.fft2_m(pad(x)).abs(); y = y + 0.05 * torch.randn_like(y); a, b = feats(y); T.append((x, a, b))
Tx = torch.cat([t[0] for t in T]); Tlm = torch.cat([t[1] for t in T]); Tac = torch.cat([t[2] for t in T])
img = lambda c: (mu + (c @ Uf).view(-1, 3, 256, 256)).clamp(0, 1)
def report(name, pred_val, pred_test):
    v = [ps(img(pred_val[i:i + 1]), img(C[ntr + i:ntr + i + 1])) for i in range(pred_val.shape[0])]      # vs the sample's own projection
    t = [ps(img(pred_test[i:i + 1]), Tx[i:i + 1]) for i in range(len(files))]
    print(f"   {name:44s} held-out model samples (vs their projection) {np.mean(v):6.2f} dB | 12 REAL faces {np.mean(t):6.2f} dB (worst {np.min(t):.1f}, best {np.max(t):.1f}, below 14 dB: {int((np.array(t) < 14).sum())})", flush=True)
    return np.mean(t)
ceil = [ps(img(((Tx[i:i + 1] - mu).flatten(1) @ Uf.T)), Tx[i:i + 1]) for i in range(len(files))]
print(f"{ntr} training samples | ceiling on the 12 real faces {np.mean(ceil):.2f} dB | mean face {np.mean([ps(mu, Tx[i:i+1]) for i in range(len(files))]):.2f} dB | eigenface magnitude fit (earlier) 16.8 dB")
best = (-1, None)
for fname, Ftr, Fte in [("low-frequency log-magnitudes", lm, Tlm), ("short-lag autocorrelation", ac, Tac), ("both", torch.cat([lm, ac], 1), torch.cat([Tlm, Tac], 1))]:
    m_, s_ = Ftr[:ntr].mean(0), Ftr[:ntr].std(0).clamp(min=1e-6); X = (Ftr - m_) / s_; Xt = (Fte - m_) / s_; Xtr, Xva = X[:ntr], X[ntr:]
    K = Xtr @ Xtr.T                                            # dual ridge (features > samples)
    for lam in (1e2, 1e3, 1e4, 1e5):
        al = torch.linalg.solve(K + lam * torch.eye(ntr, device=dev), C[:ntr]); W = Xtr.T @ al
        r = report(f"ridge, {fname}, lambda {lam:g}", Xva @ W, Xt @ W)
        if r > best[0]: best = (r, {"W": W.cpu(), "m": m_.cpu(), "s": s_.cpu(), "feat": fname, "kind": "ridge"})
    # kernel ridge (Gaussian kernel) on the 256 leading directions of the features
    Uu, Ss, Vv = torch.pca_lowrank(Xtr, q=256, niter=4); Z = Xtr @ Vv; Zva = Xva @ Vv; Zt = Xt @ Vv; zs = Z.std(0); Z, Zva, Zt = Z / zs, Zva / zs, Zt / zs
    d2 = torch.cdist(Z, Z) ** 2; med = d2.median()
    for g_ in (0.5, 1.0, 2.0):
        for lam in (1e-2, 1e-1, 1.0):
            Kk = torch.exp(-d2 / (g_ * med)); al = torch.linalg.solve(Kk + lam * torch.eye(ntr, device=dev), C[:ntr])
            pv = torch.exp(-torch.cdist(Zva, Z) ** 2 / (g_ * med)) @ al; pt = torch.exp(-torch.cdist(Zt, Z) ** 2 / (g_ * med)) @ al
            r = report(f"kernel ridge, {fname}, width {g_}, lambda {lam:g}", pv, pt)
            if r > best[0]: best = (r, {"kind": "krr", "feat": fname, "m": m_.cpu(), "s": s_.cpu(), "V": Vv.cpu(), "zs": zs.cpu(), "Z": Z.cpu(), "al": al.cpu(), "gm": float(g_ * med)})
print("best on the real faces:", best[1]["kind"], best[1]["feat"], f"{best[0]:.2f} dB"); torch.save(best[1], "../mode_probe/reg_best.pt")
