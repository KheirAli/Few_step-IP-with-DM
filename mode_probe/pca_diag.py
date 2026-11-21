"""Why does the eigenface fit lose 6 dB?  Same basis and same optimiser on three kinds of measurement (run from PnP-CM root).
A  a random face that lies EXACTLY in the subspace           -> pure optimisation difficulty
B  the oracle projection of a real test face (in subspace)   -> realistic coefficients, still no model mismatch
C  the real test face (partly outside the subspace)          -> the actual problem
Also: start the fit at the oracle coefficients (diagnostic) and compare the loss values of the two solutions."""
import sys, os, glob, re, numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
files = sorted(glob.glob("../mode_probe/data_paper/celeba_hq/celeba_hq/*.png"), key=nat)[:12]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
A = lambda img: FU.fft2_m(pad(img)).abs()
def img_of(c, k): return mu + (c @ U[:k].flatten(1)).view(1, 3, 256, 256)
def loss_of(c, k, y): return float(((A(img_of(c, k)) - y) ** 2).mean())
def fit(y, K, c0=None):
    c = torch.zeros(0, device=dev) if c0 is None else c0.clone(); ks = [k for k in (8, 16, 32, 64, 128, 256, 512) if k <= K] if c0 is None else [K]
    for k in ks:
        Uk = U[:k].flatten(1); c = torch.cat([c, torch.zeros(k - c.numel(), device=dev)]).detach().requires_grad_(True)
        opt = torch.optim.LBFGS([c], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = ((A(mu + (c @ Uk).view(1, 3, 256, 256)) - y) ** 2).mean() + 1e-4 * (c ** 2 / var[:k]).mean(); l.backward(); return l
        opt.step(closure)
    return c.detach()
torch.manual_seed(0)
for K in (64, 256):
    R = {}
    add = lambda key, v: R.setdefault(key, []).append(v)
    for f in files:
        x = ld(f); cstar = (x - mu).flatten() @ U[:K].flatten(1).T; xB = img_of(cstar, K); cA = torch.randn(K, device=dev) * var[:K].sqrt(); xA = img_of(cA, K)
        for tag, xt in (("A", xA), ("B", xB)):
            for sig in (0.0, 0.05):
                y = A(xt) + sig * torch.randn(1, 3, 384, 384, device=dev); c = fit(y, K); add(f"{tag} noise {sig}", ps(img_of(c, K), xt))
        y = A(x) + 0.05 * torch.randn(1, 3, 384, 384, device=dev); c = fit(y, K); co = fit(y, K, c0=cstar)
        add("C vs truth", ps(img_of(c, K).clamp(0, 1), x)); add("C vs oracle projection", ps(img_of(c, K), xB)); add("C ceiling (oracle projection vs truth)", ps(xB.clamp(0, 1), x))
        add("C started AT oracle coefficients, vs truth", ps(img_of(co, K).clamp(0, 1), x))
        add("loss at oracle coefficients", loss_of(cstar, K, y)); add("loss at fitted solution", loss_of(c, K, y)); add("loss after fit started at oracle", loss_of(co, K, y))
        yn = A(x); add("share of magnitude energy from the part outside the subspace", float(((yn - A(xB)) ** 2).sum() / (yn ** 2).sum()))
    print(f"--- {K} components, 12 images ---")
    for k_, v in R.items(): print(f"   {k_:62s} mean {np.mean(v):8.4f}   worst {np.min(v):8.4f}   best {np.max(v):8.4f}")
