"""Draw unconditional samples from the consistency model (2 calls each) and store them as uint8.  Run from PnP-CM root."""
import sys, os, yaml, torch
from argparse import Namespace
sys.path.insert(0, ".")
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition, append_dims
dev = torch.device("cuda:0"); sh = int(sys.argv[1]); N = int(sys.argv[2])
def ns(d):
    n = Namespace()
    for k, v in d.items(): setattr(n, k, ns(v) if isinstance(v, dict) else v)
    return n
config = ns(yaml.safe_load(open("configs/celeba_hq_256.yml"))); md = dict(vars(config.model)); md["image_size"] = config.data.image_size
model = create_model(**md); model.load_state_dict(torch.load("exp/logs/celeba_hq/ema_0.9999432189950708_1175000.pt", map_location=dev)); model.to(dev).eval()
def cm(z, L):
    L = torch.tensor([L], device=dev); cs, co, ci = [append_dims(s, z.ndim) for s in get_scalings_for_boundary_condition(L)]
    return torch.clamp(co * model(ci * z, (1000 * 0.25 * torch.log(L + 1e-44)).expand(z.shape[0])) + cs * z, -1, 1)
torch.manual_seed(90000 + sh); X = []
with torch.no_grad():
    for _ in range(N // 32):
        x = cm(80.0 * torch.randn(32, 3, 256, 256, device=dev), 80.0); x = (cm(x + 0.8 * torch.randn_like(x), 0.8) + 1) / 2
        X.append((x * 255).round().to(torch.uint8).cpu())
torch.save(torch.cat(X), f"../mode_probe/bank64k/part_{sh}.pt"); print("done", sh)
