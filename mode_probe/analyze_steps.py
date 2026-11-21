"""Average the intermediate denoised images of ONE run (no extra cost).
last-k = mean of the last k denoised images of the final pass.  Phase retrieval: PSNR with the sign known."""
import glob, itertools, re, sys
import numpy as np, torch, lpips
from PIL import Image
torch.set_num_threads(4)
dev = torch.device("cuda:0"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)

def score(x, gt, pr):
    c = (F * x) if pr else x
    ps = [float(10 * torch.log10(1 / (((k + 1) / 2 - (gt + 1) / 2) ** 2).mean())) for k in c]; k = int(np.argmax(ps))
    with torch.no_grad(): l = float(lp(c[k:k+1].to(dev), gt[None].to(dev)))
    return ps[k], l

for t in ["sr_bicubic", "deblur_gauss", "inpainting", "deblur_nl", "jpeg", "phase_retrieval"]:
    pr = t == "phase_retrieval"
    for name, v in [("baseline", "stpb"), ("ours", "stp")]:
        res = {}; n = 0
        for d in glob.glob(f"../mode_probe/exp_fin_{v}_{t}_s*"):
            for rf in glob.glob(f"{d}/raw_*.pt"):
                i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
                if not lab: continue
                raw = torch.load(rf)
                if not raw.get("steps"): continue
                gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
                lastpass = max(p for p, _, _ in raw["steps"])
                xs = [x.float() for p, _, x in raw["steps"] if p == lastpass]
                allx = [x.float() for _, _, x in raw["steps"]]
                n += 1
                for k in range(1, len(xs) + 1):
                    res.setdefault(f"last{k}", []).append(score(torch.stack(xs[-k:]).mean(0), gt, pr))
                if len(allx) > len(xs):
                    res.setdefault("both passes", []).append(score(torch.stack(allx).mean(0), gt, pr))
                # weights growing linearly toward the last step
                w = torch.arange(1, len(xs) + 1).float(); w = w / w.sum()
                res.setdefault("linear weights", []).append(score((torch.stack(xs) * w.view(-1, 1, 1, 1, 1)).sum(0), gt, pr))
        if not n: print(f"{t:16s} {name:8s} no data"); continue
        print(f"{t:16s} {name:8s} n={n:3d} | " + " | ".join(f"{k}: {np.mean([a for a, _ in r]):.2f}/{np.mean([b for _, b in r]):.3f}" for k, r in res.items()), flush=True)
