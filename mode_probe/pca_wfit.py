"""Eigenface fit with the out-of-subspace detail treated as frequency-dependent NOISE (run from PnP-CM root).
From fresh samples of the consistency model, measure for every Fourier coefficient how much the magnitude changes when a face is
replaced by its k-component projection.  Frequencies where the subspace explains the magnitude badly get a small weight."""
import sys, os, glob, re, yaml, numpy as np, torch
from argparse import Namespace
from PIL import Image
sys.path.insert(0, "."); import functions.fft_util as FU
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition, append_dims
dev = torch.device("cuda:0"); B_ = torch.load("../mode_probe/pca_cm.pt"); mu, U, var = B_["mu"].to(dev), B_["U"].to(dev), B_["var"].to(dev)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); A = lambda img: FU.fft2_m(pad(img)).abs(); ps = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
wf = "../mode_probe/pca_resvar.pt"
if not os.path.exists(wf):
    def ns(d):
        n = Namespace()
        for k, v in d.items(): setattr(n, k, ns(v) if isinstance(v, dict) else v)
        return n
    config = ns(yaml.safe_load(open("configs/celeba_hq_256.yml"))); md = dict(vars(config.model)); md["image_size"] = config.data.image_size
    model = create_model(**md); model.load_state_dict(torch.load("exp/logs/celeba_hq/ema_0.9999432189950708_1175000.pt", map_location=dev)); model.to(dev).eval()
    def cm(z, L):
        L = torch.tensor([L], device=dev); cs, co, ci = [append_dims(s, z.ndim) for s in get_scalings_for_boundary_condition(L)]
        return torch.clamp(co * model(ci * z, (1000 * 0.25 * torch.log(L + 1e-44)).expand(z.shape[0])) + cs * z, -1, 1)
    torch.manual_seed(123); acc = {k: 0 for k in (64, 256, 512)}; n = 0
    with torch.no_grad():
        for _ in range(16):                                    # 256 held-out samples (different seed from the basis)
            x = cm(80.0 * torch.randn(16, 3, 256, 256, device=dev), 80.0); x = (cm(x + 0.8 * torch.randn_like(x), 0.8) + 1) / 2; n += 16
            for k in acc:
                xp = mu + (((x - mu).flatten(1) @ U[:k].flatten(1).T) @ U[:k].flatten(1)).view(-1, 3, 256, 256)
                acc[k] = acc[k] + ((A(x) - A(xp)) ** 2).sum(0, keepdim=True)
    torch.save({k: (v / n).cpu() for k, v in acc.items()}, wf); del model
RV = torch.load(wf)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
files = sorted(glob.glob("../mode_probe/data_paper/celeba_hq/celeba_hq/*.png"), key=nat)[:int(os.environ.get("NIMG", "12"))]
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
def fit(y, K, W, c_init=None):
    c = torch.zeros(0, device=dev) if c_init is None else c_init.clone(); ks = [k for k in (8, 16, 32, 64, 128, 256, 512) if k <= K]
    for k in ks:
        Uk = U[:k].flatten(1); c = torch.cat([c, torch.zeros(max(0, k - c.numel()), device=dev)])[:max(k, c.numel())].detach().requires_grad_(True)
        if c.numel() > k: continue
        opt = torch.optim.LBFGS([c], max_iter=150, history_size=30, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = (W * (A(mu + (c @ Uk).view(1, 3, 256, 256)) - y) ** 2).mean() + 1e-4 * (c ** 2 / var[:k]).mean(); l.backward(); return l
        opt.step(closure)
    return c.detach(), float((W * (A(mu + (c.detach() @ U[:K].flatten(1)).view(1, 3, 256, 256)) - y) ** 2).mean())
torch.manual_seed(1); out = {}
for K in (64, 256, 512):
    rv = RV[K].to(dev); Wn = 1 / (0.05 ** 2 + rv); Wn = Wn / Wn.mean(); W1 = torch.ones_like(Wn); res = {"plain": [], "weighted": [], "weighted, best of 4 starts": [], "ceiling": []}
    for f in files:
        x = ld(f); y = A(x) + 0.05 * torch.randn(1, 3, 384, 384, device=dev)
        cstar = (x - mu).flatten() @ U[:K].flatten(1).T; img = lambda c: (mu + (c @ U[:K].flatten(1)).view(1, 3, 256, 256)).clamp(0, 1)
        res["ceiling"].append(ps(img(cstar), x)); res["plain"].append(ps(img(fit(y, K, W1)[0]), x))
        c1, l1 = fit(y, K, Wn); res["weighted"].append(ps(img(c1), x)); best = (l1, c1)
        for s_ in range(3):
            c0 = torch.randn(8, device=dev) * var[:8].sqrt(); cs_, ls_ = fit(y, K, Wn, c_init=c0)
            if ls_ < best[0]: best = (ls_, cs_)
        res["weighted, best of 4 starts"].append(ps(img(best[1]), x)); out[(K, os.path.basename(f))] = best[1].cpu()
    print(f"--- {K} components, {len(files)} images (PSNR vs the true image) ---")
    for k_, v in res.items(): print(f"   {k_:28s} mean {np.mean(v):6.2f}   worst {np.min(v):6.2f}   best {np.max(v):6.2f}   images below 14 dB: {int((np.array(v) < 14).sum())}")
torch.save(out, "../mode_probe/pca_wfit_coeffs.pt")
