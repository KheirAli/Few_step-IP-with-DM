"""Single run vs average of two runs (same measurement, different sampler noise).
Phase retrieval: PSNR with the sign known; before averaging, run 2 is sign-aligned to run 1 per channel
using only the two outputs."""
import glob, itertools, re, json
import numpy as np, torch, lpips
from PIL import Image
torch.set_num_threads(4)
dev = torch.device("cuda:0"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
TASKS = ["sr_bicubic", "deblur_gauss", "inpainting", "deblur_nl", "jpeg", "phase_retrieval"]

def load(v, t):
    out = {}
    for d in glob.glob(f"../mode_probe/exp_fin_{v}_{t}_s*"):
        for rf in glob.glob(f"{d}/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if lab:
                gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
                out[i] = (torch.load(rf)["x_raw"], gt)
    return out

def score(x, gt, pr):
    c = (F * x) if pr else x
    ps = [float(10 * torch.log10(1 / (((k + 1) / 2 - (gt + 1) / 2) ** 2).mean())) for k in c]; k = int(np.argmax(ps))
    with torch.no_grad(): l = float(lp(c[k:k+1].to(dev), gt[None].to(dev)))
    return ps[k], l

for t in TASKS:
    pr = t == "phase_retrieval"
    for name, v in [("baseline", "bs"), ("unified+shaped", "shs")]:
        a, b = load(v + "1", t), load(v + "2", t); ids = sorted(set(a) & set(b))
        if not ids: print(t, name, "no data"); continue
        s1, s2, av, flips = [], [], [], 0
        for i in ids:
            x1, gt = a[i]; x2 = b[i][0]
            if pr:
                sg = torch.sign((x1 * x2).sum(dim=(2, 3), keepdim=True)); flips += int((sg < 0).any()); x2 = x2 * sg
            s1.append(score(x1, gt, pr)); s2.append(score(x2, gt, pr)); av.append(score((x1 + x2) / 2, gt, pr))
        s = np.array(s1 + s2); av = np.array(av)
        print(f"{t:16s} {name:15s} n={len(ids):3d} | single PSNR {s[:,0].mean():5.2f} LPIPS {s[:,1].mean():.3f} | avg of 2 PSNR {av[:,0].mean():5.2f}±{av[:,0].std()/np.sqrt(len(ids)):.2f} LPIPS {av[:,1].mean():.3f}"
              + (f" | sign differed between runs on {flips}" if pr else ""), flush=True)
