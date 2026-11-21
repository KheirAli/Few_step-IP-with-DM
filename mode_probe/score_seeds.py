"""Score launch_seeds.sh and print a markdown table (run from PnP-CM/):  python ../mode_probe/score_seeds.py >> ../server1_results.md"""
import glob, re, itertools, numpy as np, torch, lpips
from PIL import Image
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
REF = {"sr_bicubic": "27.96/0.285", "deblur_gauss": "29.50/0.246", "inpainting": "29.87/0.199", "jpeg": "26.77/0.354", "deblur_nl": "26.31/0.335", "hdr": "25.22/0.307", "quant": "30.87/0.216", "onebit": "29.39/0.239", "prmask": "26.78/0.392"}
DEG = {"sr_bicubic": "sr_bicubic", "deblur_gauss": "deblur_gauss", "inpainting": "inpainting", "jpeg": "jpeg", "deblur_nl": "deblur_nl", "hdr": "deblur_nl", "quant": "deblur_nl", "onebit": "deblur_nl", "prmask": "phase_retrieval"}
def score(pat, pr):
    R = {}
    for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab: continue
            gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float()
            if pr: c = F * x; x = c[int(((c - gt) ** 2).mean(dim=(1, 2, 3)).argmin())]     # masked phase retrieval: channel signs fixed with the truth (as in the reference)
            with torch.no_grad(): R[i] = (float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean())), float(lp(x[None].to(dev), gt[None].to(dev))))
    return R
print("\n## Job 1: three sampler seeds, 107 test images, 4 calls (PSNR / LPIPS)\n\n| Task | Reference (seed 1, original server) | Seed 1 | Seed 2 | Seed 3 | Per-image sd across seeds (dB) | Images |\n|---|---|---|---|---|---|---|")
for t, deg in DEG.items():
    S = [score(f"seed{s}_{t}_{deg}", t == "prmask") for s in (1, 2, 3)]; cells = [f"{np.mean([v[0] for v in r.values()]):.2f}/{np.mean([v[1] for v in r.values()]):.3f}" if r else "missing" for r in S]
    com = set.intersection(*[set(r) for r in S]) if all(S) else set(); sd = f"{np.mean([np.std([r[i][0] for r in S]) for i in com]):.2f}" if com else "n/a"
    print(f"| {t} | {REF[t]} | " + " | ".join(cells) + f" | {sd} | {'/'.join(str(len(r)) for r in S)} |", flush=True)
