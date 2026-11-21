"""Do failures happen, and are they stochastic?  107 test images, ours (4 sampler seeds), PnP-CM, DAPS (matched noise).
Failure types: (a) heavy tail of our own scores, (b) seed-dependent outcomes, (c) images where another method is much better."""
import glob, itertools, re, json, os, numpy as np, torch, lpips
from PIL import Image
dev = torch.device("cuda:0"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev); L = "../mode_probe"
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
def load(pat):
    R = {}
    for d in glob.glob(f"{L}/exp_fin_{pat}_s*"):
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if lab: R[i] = (torch.load(rf)["x_raw"][0].float(), lab[0])
    return R
def sc(x, gt, flip=False):
    if flip: c = F * x; x = c[int(((c - gt) ** 2).mean(dim=(1, 2, 3)).argmin())]
    with torch.no_grad(): return float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean())), float(lp(x[None].to(dev), gt[None].to(dev)))
# DAPS outputs are in lexicographic file order; match them to our images by content of the ground truth
DD = "/usr/scratch/akheirandish3/PNP/baselines_src/DAPS"; dfiles = sorted(glob.glob(DD + "/dataset/paper-celeba/*.png")); dgt = torch.stack([ld(f) for f in dfiles])
def daps_for(gt, task):
    j = int(((dgt - gt[None]) ** 2).mean(dim=(1, 2, 3)).argmin()); f = f"{DD}/results_paper2/{task}/samples/{j:05d}_run0000.png"
    return ld(f) if os.path.exists(f) else None
TASKS = [("sr_bicubic", "sr_bicubic", "pvbase_sr_bicubic", "down_sampling_n01"), ("deblur_gauss", "deblur_gauss", "pvbase_deblur_gauss", "gaussian_blur_n01"), ("inpainting", "inpainting", "pvbase_inpainting", "inpainting_rand_n01"),
         ("jpegtrue", "jpeg", "pvbase_jpeg", "jpeg5"), ("deblur_nl", "deblur_nl", "pvbase_deblur_nl", "nonlinear_blur_n01"), ("hdr", "deblur_nl", "pvhdrb_deblur_nl", "hdr_n01"),
         ("quant", "deblur_nl", "pvquantb_deblur_nl", "quant8"), ("onebit", "deblur_nl", "pvonebitb_deblur_nl", "onebit"), ("prmask", "phase_retrieval", "pvbase_phase_retrieval", "pr_masked")]
out = {}
for task, deg, bpat, dname in TASKS:
    runs = [load(f"s8test_{task}_ship4_{deg}")] + [load(f"par{sd}_{task}_{deg}") for sd in (2, 3, 4)]; base = load(bpat); ids = sorted(set.intersection(*[set(r) for r in runs])); pr = task == "prmask"; rows = []
    for i in ids:
        gt = ld(runs[0][i][1]); s = [sc(r[i][0], gt, pr) for r in runs]; b = sc(base[i][0], gt, pr) if i in base else (np.nan, np.nan); dx = daps_for(gt, dname); d = sc(dx, gt, pr) if dx is not None else (np.nan, np.nan)
        rows.append([a for a, _ in s] + [l for _, l in s] + list(b) + list(d))
    A = np.array(rows); out[task] = A.tolist(); P, Lp = A[:, :4], A[:, 4:8]; p1, l1 = P[:, 0], Lp[:, 0]; bp, bl, dp, dl = A[:, 8], A[:, 9], A[:, 10], A[:, 11]
    tail = l1 >= np.percentile(l1, 90)
    print(f"{task:12s} n={len(ids)} | ours PSNR median {np.median(p1):.2f}, 5th pct {np.percentile(p1, 5):.2f}, worst {p1.min():.2f} | LPIPS median {np.median(l1):.3f}, 95th pct {np.percentile(l1, 95):.3f}, worst {l1.max():.3f}"
          f" | seed range >1 dB on {(np.ptp(P, 1) > 1).mean() * 100:.0f}%, LPIPS range >0.05 on {(np.ptp(Lp, 1) > 0.05).mean() * 100:.0f}%"
          f" | PnP-CM better by >1 dB on {np.nanmean(bp - p1 > 1) * 100:.0f}%, by >0.05 LPIPS on {np.nanmean(l1 - bl > 0.05) * 100:.0f}%"
          f" | DAPS better by >1 dB on {np.nanmean(dp - p1 > 1) * 100:.0f}%, by >0.05 LPIPS on {np.nanmean(l1 - dl > 0.05) * 100:.0f}%"
          f" | worst-10% LPIPS images: ours {l1[tail].mean():.3f}, best of 4 seeds {Lp[tail].min(1).mean():.3f}, PnP-CM {np.nanmean(bl[tail]):.3f}, DAPS {np.nanmean(dl[tail]):.3f}", flush=True)
json.dump(out, open(f"{L}/failure_scan.json", "w"))
