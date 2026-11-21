"""Local face subspaces and the constant mismatch correction for unmasked phase retrieval (run from PnP-CM root).
A bank of 8192 faces sampled from the consistency model is generated once (stored in half precision).
Part 1  constant correction: add the AVERAGE mismatch (over the bank) to the eigenface magnitude model and refit.
Part 2  local subspaces: for each test face take its 200 nearest bank faces and build a subspace from them.  Neighbours chosen
        (a) by the true image (oracle, diagnostic), (b) by our measurement-only coarse fit, (c) by low-frequency magnitudes.
        Reported: best possible image PSNR at 64 / 128 components, and how well the subspace reproduces the true magnitudes."""
import sys, os, glob, re, yaml, numpy as np, torch
from argparse import Namespace
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); A = lambda im: FU.fft2_m(pad(im)).abs(); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
bank = "../mode_probe/face_bank_8192.pt"
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
    torch.manual_seed(2024); X = []; LM = []; rb = 0
    with torch.no_grad():
        for _ in range(512):
            x = cm(80.0 * torch.randn(16, 3, 256, 256, device=dev), 80.0); x = (cm(x + 0.8 * torch.randn_like(x), 0.8) + 1) / 2
            a = A(x); c = (x - mu).flatten(1) @ Uf.T; rb = rb + (a - A(mu + (c @ Uf).view(-1, 3, 256, 256))).sum(0, keepdim=True)
            X.append(x.half().cpu()); LM.append(torch.log1p(a)[..., 160:224, 160:224].flatten(1).half().cpu())
    torch.save({"x": torch.cat(X), "lm": torch.cat(LM), "rbar": (rb / 8192).cpu()}, bank); del model
D = torch.load(bank); XB = D["x"].to(dev).flatten(1); LMB = D["lm"].float().to(dev); rbar = D["rbar"].to(dev)     # XB half precision [8192, 196608]
R = torch.load("../mode_probe/reg_best.pt"); W, m_, s_ = R["W"].to(dev), R["m"].to(dev), R["s"].to(dev)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
files = sorted(glob.glob("../mode_probe/data_paper/celeba_hq/celeba_hq/*.png"), key=nat)[:int(os.environ.get("NIMG", "107"))]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
def fit(y, c0, corr):
    c = c0.clone()
    for k in (64, 128, 256, 512):
        ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = ((A(mu + (ck @ Uf[:k]).view(1, 3, 256, 256)) + corr - y) ** 2).mean() + 1e-4 * (ck ** 2 / var[:k]).mean(); l.backward(); return l
        opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
    return c
def dist(xq):                                                  # squared distance to every bank face, in full precision (half precision overflowed)
    q = xq.flatten(1).float(); return torch.cat([((XB[i:i + 512].float() - q) ** 2).sum(1) for i in range(0, XB.shape[0], 512)])
def local(xq, nn=200):                                         # subspace from the nn bank faces nearest to the query image xq [1,3,256,256]
    d = dist(xq); idx = torch.topk(d, nn, largest=False).indices; Z = XB[idx].float(); m = Z.mean(0, keepdim=True); Zc = Z - m
    ev, V = torch.linalg.eigh(Zc @ Zc.T); o = torch.argsort(ev, descending=True)[:128]; Ul = (Zc.T @ V[:, o]) / ev[o].clamp(min=1e-8).sqrt(); return m, Ul.T
def proj(x, m, Ul, k): return (m + ((x.flatten(1) - m) @ Ul[:k].T) @ Ul[:k]).view(1, 3, 256, 256)
img = lambda c: (mu + (c @ Uf).view(1, 3, 256, 256)).clamp(0, 1)
res = {}; add = lambda k_, v: res.setdefault(k_, []).append(v)
torch.manual_seed(5)
for f in files:
    x = ld(f); a = A(x); y = a + 0.05 * torch.randn_like(a)
    creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]
    c_plain = fit(y, creg, 0.0); c_corr = fit(y, creg, rbar); add("fit, plain model", ps(img(c_plain), x)); add("fit, model + average mismatch", ps(img(c_corr), x))
    cstar = (x - mu).flatten() @ Uf.T
    for k in (64, 128):
        gp = (mu + (cstar[:k] @ Uf[:k]).view(1, 3, 256, 256)); add(f"global {k}: image ceiling", ps(gp.clamp(0, 1), x)); add(f"global {k}: magnitude mismatch", float((a - A(gp)).norm() / a.norm()))
    lmq = torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1); il = torch.topk(torch.cdist(lmq, LMB)[0], 200, largest=False).indices
    for tag, q in (("oracle neighbours (true image)", x), ("neighbours of our coarse fit", img(c_plain))):
        m, Ul = local(q)
        for k in (64, 128):
            lp_ = proj(x, m, Ul, k); add(f"local {k}, {tag}: image ceiling", ps(lp_.clamp(0, 1), x)); add(f"local {k}, {tag}: magnitude mismatch", float((a - A(lp_)).norm() / a.norm()))
    Z = XB[il].float(); m = Z.mean(0, keepdim=True); Zc = Z - m; ev, V = torch.linalg.eigh(Zc @ Zc.T); o = torch.argsort(ev, descending=True)[:128]; Ul = ((Zc.T @ V[:, o]) / ev[o].clamp(min=1e-8).sqrt()).T
    for k in (64, 128):
        lp_ = proj(x, m, Ul, k); add(f"local {k}, neighbours by low-frequency magnitudes: image ceiling", ps(lp_.clamp(0, 1), x)); add(f"local {k}, neighbours by low-frequency magnitudes: magnitude mismatch", float((a - A(lp_)).norm() / a.norm()))
    d = dist(x); add("nearest single bank face to the true image (PSNR)", ps(XB[d.argmin()].float().view(1, 3, 256, 256), x))
    for nn_ in (50, 1000):
        m, Ul = local(x, nn_); kk = min(64, nn_ - 1); lp_ = proj(x, m, Ul, kk); add(f"local {kk}, oracle neighbours, {nn_} neighbours: image ceiling", ps(lp_.clamp(0, 1), x))
print(f"{len(files)} images")
for k_, v in res.items():
    v = np.array(v); print(f"   {k_:74s} mean {v.mean():7.3f}   5th pct {np.percentile(v, 5):7.3f}   worst {v.min():7.3f}" + (f"   below 14 dB: {int((v < 14).sum())}" if "fit," in k_ else ""))
