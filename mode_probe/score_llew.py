"""Score launch_llew.sh as a markdown table (run from PnP-CM/):  python ../mode_probe/score_llew.py >> ../server1_results.md     env: TAG (llew), NIMG (107)"""
import glob, re, os, numpy as np, torch, lpips
from PIL import Image
TAG = os.environ.get("TAG", "llew"); NIMG = int(os.environ.get("NIMG", "107")); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
def score(pat):
    P = []; Lp = []
    for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab: continue
            gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float()
            with torch.no_grad(): P.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Lp.append(float(lp(x[None].to(dev), gt[None].to(dev))))
    return (np.mean(P), np.mean(Lp), len(P)) if P else None
NAMES = {"blk8": "8x8 blocks", "pixs": "per pixel", "pix": "per pixel and colour", "dct30": "30 smooth maps", "eig30": "30-dim embedding", "band8": "8 frequency bands", "gate": "gated by step-to-step change", "zscalar": "scalar + data-fit term", "zband8": "8 bands + data-fit term"}
M = [(m, NAMES.get(m, m)) for m in os.environ.get("MODES", "blk8 pixs pix dct30 eig30").split() if m != "scalar"]; TITLE = os.environ.get("TITLE", "Job 2: structured extrapolation weights")
print(f"\n## {TITLE}, {NIMG} test images, 4 calls, calibrated on generated images\n\nPSNR / LPIPS; in brackets the PSNR change against scalar weights. Masked phase retrieval is scored as produced (sign rule on).\n\n| Task | Scalar (9 numbers) | " + " | ".join(n for _, n in M) + " |\n|---|---|" + "---|" * len(M))
for t, deg in [("sr_bicubic", "sr_bicubic"), ("deblur_gauss", "deblur_gauss"), ("inpainting", "inpainting"), ("jpeg", "jpeg"), ("jpegtrue", "jpeg"), ("deblur_nl", "deblur_nl"), ("hdr", "deblur_nl"), ("quant", "deblur_nl"), ("onebit", "deblur_nl"), ("prmask", "phase_retrieval")]:
    s = score(f"{TAG}test_scalar_{t}_{deg}")
    if not s: continue
    row = f"| {t} | {s[0]:.2f}/{s[1]:.3f}" + ("" if s[2] == NIMG else f" (n={s[2]})")
    for m, _ in M:
        r = score(f"{TAG}test_{m}_{t}_{deg}"); row += " | " + (f"{r[0]:.2f}/{r[1]:.3f} ({r[0] - s[0]:+.2f})" + ("" if r[2] == NIMG else f" (n={r[2]})") if r else "missing")
    print(row + " |", flush=True)
