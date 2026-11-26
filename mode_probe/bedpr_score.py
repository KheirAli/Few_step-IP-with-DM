"""Score the bedroom unmasked phase-retrieval test runs exactly as pr_today.py / attempt2.py score the face runs: PSNR of the clamped raw output against
the saved ground-truth PNG and LPIPS (VGG), every image x every noise draw, NO allowance for the 180-degree flip (a flipped reconstruction counts as it is;
it shows up in the share below 14 dB).  The share of runs whose flipped output would score higher is printed for information.  usage (from PnP-CM/): bedpr_score.py"""
import glob, re, json, os, numpy as np, torch, lpips
from PIL import Image
exec(open("../mode_probe/bedpr_cfg.py").read()); lp = lpips.LPIPS(net="vgg", verbose=False).cuda(); out = {}
for t, k, c, n, desc in sorted(JOBS, key=lambda j: (j[0][5], j[3])):
    Ps = []; Ls = []; Fl = []; per = {}
    for d in sorted(glob.glob(f"{L}/exp_fin_{t}m[012]_zband8_{k}_s0")):
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1; x = torch.load(rf)["x_raw"].float().clamp(-1, 1)
            with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x.cuda(), gt.cuda()))); Fl.append(float(10 * torch.log10(1 / (((torch.flip(x, dims=(2, 3)) - gt) / 2) ** 2).mean())) > Ps[-1])
        per[d[-5]] = len(Ps)
    Ps = np.array(Ps)
    if len(Ps): out[t] = dict(psnr=float(Ps.mean()), lpips=float(np.mean(Ls)), n=len(Ps), below14=float(100 * (Ps < 14).mean()), flipped_better=float(100 * np.mean(Fl)), median=float(np.median(Ps)))
    print(f"BEDPR TEST {desc}, {n} calls [{t}]: {Ps.mean():.2f}/{np.mean(Ls):.3f} (n={len(Ps)}), below 14 dB {100 * (Ps < 14).mean():.1f}% | median {np.median(Ps):.2f} | flipped output closer to the truth in {100 * np.mean(Fl):.1f}% of runs" if len(Ps) else f"BEDPR TEST {t} missing", flush=True)
json.dump(out, open(f"{L}/bedpr_results.json", "w"), indent=1)
