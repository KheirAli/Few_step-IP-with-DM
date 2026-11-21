"""Regression-anchored eigenface fitting for unmasked phase retrieval (run from PnP-CM root).  Same measurements for every variant.
  LO, HI: image range;  results appended to ../mode_probe/anchor_fit_<LO>.pt"""
import sys, os, glob, re, numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev); Uf = U.flatten(1)
R = torch.load("../mode_probe/reg_best.pt"); W, m_, s_ = R["W"].to(dev), R["m"].to(dev), R["s"].to(dev)
D = torch.load("../mode_probe/reg_bank_8192.pt"); Xv = (D["lm"][8192:].to(dev) - m_) / s_; rs2 = ((Xv @ W - D["c"][8192:].to(dev)) ** 2).mean(0).clamp(min=1e-8)   # regression error variance per coefficient (held-out samples)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean())); A = lambda im: FU.fft2_m(pad(im)).abs()
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
LO, HI = int(os.environ.get("LO", "0")), int(os.environ.get("HI", "12"))
files = sorted(glob.glob("../mode_probe/data_paper/celeba_hq/celeba_hq/*.png"), key=nat)[LO:HI]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
img = lambda c: (mu + (c @ Uf).view(1, 3, 256, 256)).clamp(0, 1)
def fit(y, c0, creg, lams, ks=(64, 128, 256, 512)):
    c = c0.clone()
    for k, lam in zip(ks, lams):
        ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = ((A(mu + (ck @ Uf[:k]).view(1, 3, 256, 256)) - y) ** 2).mean() + 1e-4 * (ck ** 2 / var[:k]).mean() + lam * ((ck - creg[:k]) ** 2 / rs2[:k]).mean(); l.backward(); return l
        opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
    return c
V = {"regression only": None, "regression -> free fit (current)": (0, 0, 0, 0), "constant anchor 3e-4": (3e-4,) * 4, "constant anchor 1e-3": (1e-3,) * 4, "constant anchor 3e-3": (3e-3,) * 4, "constant anchor 1e-2": (1e-2,) * 4,
     "decreasing anchor 1e-2 -> 0": (1e-2, 3e-3, 1e-3, 0), "decreasing anchor 3e-3 -> 3e-4": (3e-3, 1e-3, 5e-4, 3e-4), "decreasing anchor 3e-2 -> 1e-3": (3e-2, 1e-2, 3e-3, 1e-3)}
res = {k: [] for k in V}; res["4 starts near regression, pick by loss + anchor"] = []; res["4 starts near regression, pick by loss"] = []; res["ceiling"] = []; coef = {}
for f in files:
    x = ld(f); cstar = (x - mu).flatten() @ Uf.T
    for seed in (11, 12, 13):
        g = torch.Generator(device=dev).manual_seed(seed * 1000 + nat(os.path.basename(f))[1]); y = A(x) + 0.05 * torch.randn(1, 3, 384, 384, device=dev, generator=g)
        creg = ((torch.log1p(y.clamp(min=0))[..., 160:224, 160:224].flatten(1) - m_) / s_ @ W)[0]; res["ceiling"].append(ps(img(cstar), x))
        for name, lams in V.items():
            c = creg if lams is None else fit(y, creg, creg, lams); res[name].append(ps(img(c), x))
            if name == "decreasing anchor 3e-3 -> 3e-4": coef[(os.path.basename(f), seed)] = c.cpu()
        cands = [fit(y, creg + (0 if j == 0 else 1.0) * torch.randn(512, device=dev, generator=g) * rs2.sqrt(), creg, (3e-3, 1e-3, 5e-4, 3e-4)) for j in range(4)]
        L = lambda c: float(((A(mu + (c @ Uf).view(1, 3, 256, 256)) - y) ** 2).mean()); S = lambda c: L(c) + 3e-4 * float(((c - creg) ** 2 / rs2).mean())
        res["4 starts near regression, pick by loss + anchor"].append(ps(img(min(cands, key=S)), x)); res["4 starts near regression, pick by loss"].append(ps(img(min(cands, key=L)), x))
torch.save({"res": res, "coef": coef}, f"../mode_probe/anchor_fit_{LO}.pt")
