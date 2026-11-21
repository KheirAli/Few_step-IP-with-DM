"""Average of N sampler runs (same measurement, different sampler noise), per run family.
   python ../mode_probe/analyze_nsamp.py fam=v1,v2,v3,v4 [fam2=...]      (from PnP-CM root)
Reports single-run PSNR/LPIPS, average-of-N PSNR/LPIPS, and a split of the single-run error into the part
shared by all runs (stays after averaging) and the part that differs between runs (shrinks as 1/N).
Phase retrieval: samples are sign-aligned per channel to the first one; PSNR with the sign known."""
import glob, itertools, re, sys, json
import numpy as np, torch, lpips
from PIL import Image
torch.set_num_threads(4)
dev = torch.device("cuda:0"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
TASKS = [a for a in ["sr_bicubic", "deblur_gauss", "inpainting", "deblur_nl", "jpeg", "phase_retrieval"] if not __import__("os").environ.get("ONLY") or a in __import__("os").environ["ONLY"]]
fams = dict((a.split("=")[0], a.split("=")[1].split(",")) for a in sys.argv[1:])

def load(v, t):
    out = {}
    for d in glob.glob(f"../mode_probe/exp_fin_{v}_{t}_s*"):
        for rf in glob.glob(f"{d}/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if lab:
                gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
                out[i] = (torch.load(rf)["x_raw"][0].float(), gt)
    return out

def score(x, gt, pr):
    c = (F * x) if pr else x[None]
    mse = [float((((k - gt) / 2) ** 2).mean()) for k in c]; k = int(np.argmin(mse))
    with torch.no_grad(): l = float(lp(c[k:k+1].to(dev), gt[None].to(dev)))
    return mse[k], l, c[k]

res = {}
for t in TASKS:
    pr = t == "phase_retrieval"
    for fam, vs in fams.items():
        L = [load(v, t) for v in vs]; L = [l for l in L if l]
        if not L: continue
        ids = sorted(set.intersection(*[set(l) for l in L])); N = len(L)
        if not ids: continue
        single, avg, corr = [], {n: [] for n in range(1, N + 1)}, []
        for i in ids:
            gt = L[0][i][1]; xs = [l[i][0] for l in L]
            if pr: xs = [xs[0]] + [x * torch.sign((xs[0] * x).sum(dim=(1, 2), keepdim=True)) for x in xs[1:]]
            errs = []
            for x in xs:
                m, l, xa = score(x, gt, pr); single.append((m, l)); errs.append((xa - gt).flatten())
            for n in range(1, N + 1):
                m, l, _ = score(torch.stack(xs[:n]).mean(0), gt, pr); avg[n].append((m, l))
            if N > 1:
                corr.append(np.mean([float(torch.dot(errs[a], errs[b]) / (errs[a].norm() * errs[b].norm())) for a in range(N) for b in range(a + 1, N)]))
        ps = lambda a: float(np.mean([-10 * np.log10(m) for m, _ in a])); lpv = lambda a: float(np.mean([l for _, l in a]))
        rho = float(np.mean(corr)) if corr else float("nan")
        line = f"{t:16s} {fam:8s} n={len(ids):3d} N={N} | single {ps(single):5.2f}/{lpv(single):.3f} | " + " ".join(f"avg{n} {ps(avg[n]):5.2f}/{lpv(avg[n]):.3f}" for n in range(2, N + 1))
        line += f" | shared-error fraction {rho:.2f} -> limit of averaging {ps(single) - 10 * np.log10(rho):5.2f} dB"
        print(line, flush=True)
        res[f"{t}/{fam}"] = dict(n=len(ids), N=N, single=[ps(single), lpv(single)], avg={n: [ps(avg[n]), lpv(avg[n])] for n in avg}, rho=rho,
                                 per_image={n: [-10 * np.log10(m) for m, _ in avg[n]] for n in avg})
json.dump(res, open("../mode_probe/nsamp_results.json", "w"))
