"""Nonlinear CT, full method, everything tuned on MODEL-GENERATED images (never the test set):
1. penalty (3 values) x noise schedule (one pass of coordinate search with the combination in the objective, sched_cd2.py), 48 generated images;
2. in-loop combination weights (scalar and 8 bands + data-fit term) fitted on 128 generated images;  3. the 107 test images.   Run from PnP-CM/."""
import json, subprocess, os, glob, re, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"
pr = [subprocess.Popen([P, f"{L}/sched_cd2.py", f"ctnl{K}r{r}", str(g)], env=dict(os.environ, NSTEPS="4"), stdout=open(f"{L}/logs_x/ctnlfull_cd2_{K}_{r}.log", "w"), stderr=subprocess.STDOUT)
      for g, (K, r) in enumerate([(K, r) for K in (30, 60) for r in ("0.1", "0.3", "1")])]
[p.wait() for p in pr]; best = {}
for K in (30, 60):
    R = {r: json.load(open(f"{L}/sched_cd2_ctnl{K}r{r}.json"))["4"] for r in ("0.1", "0.3", "1") if os.path.exists(f"{L}/sched_cd2_ctnl{K}r{r}.json")}
    r = max(R, key=lambda k: R[k]["best"]); best[K] = (r, R[r])
    print(f"TUNE {K} views (generated images, combination in the objective): " + " | ".join(f"penalty {k}: start {v['start'][0]:.2f} -> tuned {v['best']:.2f}" for k, v in R.items()) + f" || chosen penalty {r}, sigma " + " ".join(f"{x:.3f}" for x in R[r]["sig"]) + f", told x{R[r]['told']}", flush=True)
pr = []
for g, K in enumerate((30, 60)):
    r, b = best[K]; cfg = f"deblur_nl NEWOP=ctnl CT_K={K} RHOX={r} SIGS={','.join(f'{v:.4f}' for v in b['sig'])} SIGPOS=0,1,2,3 SIGTOLD={b['told']}"
    env = dict(os.environ, TAG=f"ctnlF{K}", KEY=f"ctnl{K}", CFGSTR=cfg, NSTEP="4", MODES="scalar zband8", CALSH="0:64 64:128", NGPU="8", PAR="4", GOFF=str(4 * g))
    pr.append(subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=env, stdout=open(f"{L}/logs_x/launch_ctnlF{K}.log", "w"), stderr=subprocess.STDOUT, cwd=L))
[p.wait() for p in pr]
dev = torch.device("cuda:0"); lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
for K in (30, 60):
    for m in ("scalar", "zband8"):
        Ps = []; Ls = []
        for rf in glob.glob(f"{L}/exp_fin_ctnlF{K}test_{m}_ctnl{K}_deblur_nl_s*/raw_*.pt"):
            d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float()
            with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].to(dev), gt[None].to(dev))))
        print(f"FINAL {K} views, {m}: {np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)})" if Ps else f"FINAL {K} views, {m}: missing", flush=True)
