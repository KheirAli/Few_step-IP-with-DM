"""Score DPS / DAPS result folders with the same PSNR and LPIPS (VGG) as ours.  usage (from PnP-CM/): python ../mode_probe/score_base.py <celeba|bedroom> <task> [task ...]
DAPS saves samples by dataset index; the ground truth of each sample is found as the nearest image of the dataset to the sample's low-pass version is NOT used -
the index order (natural sort of the file names, as in the DAPS data class) is used and checked against the DAPS log's own PSNR."""
import sys, glob, os, re, numpy as np, torch, lpips
from PIL import Image
B = "../baselines_src"; ds = sys.argv[1]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda(); ld = lambda f: torch.from_numpy(np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
def sc(pairs):
    Ps = []; Ls = []
    for x, g in pairs:
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - g) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), g[None].cuda())))
    return f"{np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)})" if Ps else "missing"
gts = sorted(glob.glob(f"{B}/DAPS/dataset/paper-{ds}/*.png")); nat = sorted(gts, key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(f))])
for t in sys.argv[2:]:
    d = f"{B}/diffusion-posterior-sampling/results_{ds}/{t}/{t}"; print(f"BASE | {ds} | {t} | DPS | " + sc([(ld(f), ld(f.replace('/recon/', '/label/'))) for f in sorted(glob.glob(f"{d}/recon/*.png"))]) + " |", flush=True)
    S = sorted(glob.glob(f"{B}/DAPS/results_{ds}/{t}/samples/*_run0000.png")); out = {}
    for name, order in (("sorted", gts), ("natural", nat)): out[name] = sc([(ld(f), ld(order[int(os.path.basename(f)[:5])])) for f in S]) if S else "missing"
    print(f"BASE | {ds} | {t} | DAPS | file order sorted: {out['sorted']} ; natural: {out['natural']} |", flush=True)
