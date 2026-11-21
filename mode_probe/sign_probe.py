"""GT-free sign-mode selection probe for PnP-CM phase retrieval.

For each image, build the 8 per-channel sign candidates (all have identical
|F(M x)|, so the measurement cannot tell them apart) and score each with the
consistency model alone:  score(x) = E_eps || f(x + sigma*eps, sigma) - x ||^2.
Ask: does argmin score pick the candidate whose sign matches the ground truth?

Run from the PnP-CM repo root:
  python ../mode_probe/sign_probe.py <run_dir> <out_json>
"""
import sys, json, itertools, yaml, argparse
import numpy as np, torch
from PIL import Image

sys.path.insert(0, ".")
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition

run_dir, out_json = sys.argv[1], sys.argv[2]
dev = torch.device("cuda:0")
torch.manual_seed(0)

cfg = yaml.safe_load(open("configs/celeba_hq_256.yml"))["model"]
cfg["image_size"] = 256
model = create_model(**cfg)
model.load_state_dict(torch.load("exp/logs/celeba_hq/ema_0.9999432189950708_1175000.pt", map_location=dev))
model.to(dev).eval()

SIGMAS = [0.1, 0.3, 0.6, 1.0, 2.0]
N_EPS = 4
FLIPS = list(itertools.product([1, -1], repeat=3))


def load(p):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).to(dev) * 2 - 1


@torch.no_grad()
def cm(x, sigma):
    s = torch.tensor(sigma, device=dev)
    c_skip, c_out, c_in = get_scalings_for_boundary_condition(s)
    t = (1000 * 0.25 * torch.log(s + 1e-44)).repeat(x.shape[0])
    return (c_out * model(c_in * x, t) + c_skip * x).clamp(-1, 1)


@torch.no_grad()
def scores(x):
    """x: (3,H,W) in [-1,1]. Returns {sigma: [8 scores]} and candidates."""
    cands = torch.stack([x * torch.tensor(f, device=dev).view(3, 1, 1) for f in FLIPS])
    out = {}
    for sg in SIGMAS:
        acc = torch.zeros(8, device=dev)
        for _ in range(N_EPS):
            eps = torch.randn_like(cands[:1]).expand_as(cands)  # same noise for all candidates
            acc += ((cm(cands + sg * eps, sg) - cands) ** 2).mean(dim=(1, 2, 3))
        out[sg] = (acc / N_EPS).tolist()
    return out, cands


results = []
for i in range(10):
    gt = load(f"{run_dir}/labels/orig_{i}.png")
    rec = load(f"{run_dir}/samples/{i}_0.png")
    for name, x in [("gt", gt), ("recon", rec)]:
        sc, cands = scores(x)
        # the correct candidate = the one with all-positive channel correlation to GT
        corr = [[float(np.corrcoef(c[k].flatten().cpu(), gt[k].flatten().cpu())[0, 1]) for k in range(3)] for c in cands]
        true_idx = int(np.argmax([sum(np.sign(c)) for c in corr]))
        row = {"img": i, "src": name, "true_idx": true_idx}
        for sg in SIGMAS:
            s = np.array(sc[sg])
            order = np.argsort(s)
            row[f"pick@{sg}"] = int(order[0])
            row[f"ok@{sg}"] = bool(order[0] == true_idx)
            row[f"rank@{sg}"] = int(np.where(order == true_idx)[0][0])
            wrong = np.delete(s, true_idx)
            row[f"margin@{sg}"] = float(wrong.min() / s[true_idx])  # >1 means correct mode wins
        results.append(row)
        print(row, flush=True)

json.dump(results, open(out_json, "w"), indent=1)
print("\n=== accuracy (picked the correct sign mode out of 8; chance = 12.5%) ===")
for name in ["gt", "recon"]:
    rs = [r for r in results if r["src"] == name]
    for sg in SIGMAS:
        acc = np.mean([r[f"ok@{sg}"] for r in rs])
        mm = np.min([r[f"margin@{sg}"] for r in rs])
        print(f"{name:6s} sigma={sg:<4} acc={acc*100:5.1f}%  worst margin={mm:.3f}")
