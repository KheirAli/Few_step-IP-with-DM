"""LSUN bedroom, fairness and exactness follow-ups (run from PnP-CM/):
(a) PnP-CM WITH scalar extrapolation (weights fitted on bedroom-generated images), as was done on CelebA-HQ;
(b) our nonlinear CT with NO alternation in the data-fit (settings tuned on CelebA-generated images, unchanged)."""
import subprocess, os, glob, re, time, itertools, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; os.environ["DSET"] = "bedroom"
PNP = {"sr_bicubic": "sr_bicubic TILT=0 LINEXACT=1", "deblur_gauss": "deblur_gauss TILT=0 LINEXACT=1", "inpainting": "inpainting TILT=0 LINEXACT=1", "jpeg": "jpeg TILT=0 SOLVER=repo", "deblur_nl": "deblur_nl TILT=0 SOLVER=repo",
       "hdr": "deblur_nl TILT=0 NEWOP=hdr SOLVER=repo", "quant": "deblur_nl TILT=0 NEWOP=quant SOLVER=repo", "onebit": "deblur_nl TILT=0 NEWOP=onebit SOLVER=repo", "prmask": "phase_retrieval TILT=0 SOLVER=repo",
       "ctnl30": "deblur_nl TILT=0 NEWOP=ctnl CT_K=30 SOLVER=repo RHOX=1 REPO_LR=3e-2", "ctnl60": "deblur_nl TILT=0 NEWOP=ctnl CT_K=60 SOLVER=repo RHOX=1 REPO_LR=3e-2",
       "gamma": "deblur_nl TILT=0 NEWOP=gamma SOLVER=repo", "blurq": "deblur_nl TILT=0 NEWOP=blurq SOLVER=repo", "blurclip": "deblur_nl TILT=0 NEWOP=blurclip SOLVER=repo"}
ONE = "RHOX=0.03 CTNL_N=1 CTNL_MU=1 SIGS=0.9000,0.4500,0.5940,0.1650 SIGPOS=0,1,2,3 SIGTOLD=1.5"
JOBS = [(f"bedpl{t}", t, c, "scalar") for t, c in PNP.items()] + [(f"bedone{K}", f"ctnl{K}", f"deblur_nl NEWOP=ctnl CT_K={K} {ONE}", "zband8") for K in (30, 60)]
run = []
for i, (tag, key, cfg, modes) in enumerate(JOBS):
    while len([p for p in run if p.poll() is None]) >= int(os.environ.get("MAXL", "8")): time.sleep(3)
    env = dict(os.environ, TAG=tag, KEY=key, CFGSTR=cfg, NSTEP="4", MODES=modes, CALSH="0:64 64:128", TESTSH="0:29 29:58", CALDATA=f"{L}/data_bedgen", TESTDATA=f"{L}/data_bed", NGPU="8", PAR="4", GOFF=str(2 * i))
    run.append(subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=env, stdout=open(f"{L}/logs_x/launch_{tag}.log", "w"), stderr=subprocess.STDOUT, cwd=L))
[p.wait() for p in run]
lp = lpips.LPIPS(net="vgg", verbose=False).cuda(); F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3))).view(8, 3, 1, 1)
for tag, key, cfg, modes in JOBS:
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{tag}test_{modes}_{key}_*_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        if key == "prmask": c = F * x; x = c[int(((c - gt) ** 2).mean(dim=(1, 2, 3)).argmin())]
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    print(f"B2 {'PnP-CM + scalar extrapolation' if tag.startswith('bedpl') else 'ours, no alternation'} | {key}: {np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)})" if Ps else f"B2 {tag} {key} missing", flush=True)
