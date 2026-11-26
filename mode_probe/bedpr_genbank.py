"""Training images for the bedroom phase predictor: 2-call samples of the bedroom consistency model, uint8 (as gen_bank.py for faces).
Run from PnP-CM/:  bedpr_genbank.py <part> [N=8192].  Seeds 190000+part (data_bedgen used 90000..90003, so no image is shared with the
generated images used for tuning / fitting)."""
import sys, os, yaml, torch
from argparse import Namespace
sys.path.insert(0, ".")
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition, append_dims
dev = torch.device("cuda:0"); sh = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192; BS = int(os.environ.get("BS", "16"))
out = f"../mode_probe/bank64k_bed/part_{sh}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
if os.path.exists(out): print("exists", sh); sys.exit(0)
def ns(d):
    n = Namespace()
    for k, v in d.items(): setattr(n, k, ns(v) if isinstance(v, dict) else v)
    return n
config = ns(yaml.safe_load(open("configs/lsun_bedroom_256.yml"))); md = dict(vars(config.model)); md["image_size"] = config.data.image_size
model = create_model(**md); model.load_state_dict(torch.load("exp/logs/lsun_bedroom/cd_bedroom256_lpips.pt", map_location=dev)); model.to(dev).eval()
def cm(z, L):
    L = torch.tensor([L], device=dev); cs, co, ci = [append_dims(s, z.ndim) for s in get_scalings_for_boundary_condition(L)]
    return torch.clamp(co * model(ci * z, (1000 * 0.25 * torch.log(L + 1e-44)).expand(z.shape[0])) + cs * z, -1, 1)
torch.manual_seed(190000 + sh); X = []
with torch.no_grad():
    for _ in range(N // BS):
        x = cm(80.0 * torch.randn(BS, 3, 256, 256, device=dev), 80.0); x = (cm(x + 0.8 * torch.randn_like(x), 0.8) + 1) / 2
        X.append((x * 255).round().to(torch.uint8).cpu())
torch.save(torch.cat(X), out + ".tmp"); os.replace(out + ".tmp", out); print("done", sh)
