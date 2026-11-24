"""ATTRIBUTION ABLATION and GENERIC DATA-FIT (run from PnP-CM/).  Same pipeline in every arm - same noise schedule, noise tilt, 4 calls, combination
weights (8 bands + data-fit term) refitted on 128 generated images for that arm - ONLY the data step differs:
  ours     : the separable closed-form data-fit used so far
  grad     : PnP-CM's gradient solver on the same objective, learning rate and iteration count (100 or 400) chosen on generated images
  generic  : black-box table data-fit (no formula for the operator), minimiser          genmean : the same, posterior mean
107 test images, once."""
import subprocess, os, glob, re, time, json, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"
T = {"hdr": "deblur_nl NEWOP=hdr HDRX=1 SIGS=3.0,0.635,0.56,0.459 SIGPOS=0,1,2,3", "quant": "deblur_nl NEWOP=quant", "onebit": "deblur_nl NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2", "gamma": "deblur_nl NEWOP=gamma",
     "blurclip": "deblur_nl NEWOP=blurclip SPLIT_EV=0.3", "ctnl30": "deblur_nl NEWOP=ctnl CT_K=30 RHOX=0.03 CTNL_N=1 CTNL_MU=1 SIGS=0.9,0.45,0.594,0.165 SIGPOS=0,1,2,3 SIGTOLD=1.5"}
def psnr_dir(pat, need):
    Ps = []
    for rf in glob.glob(f"{L}/exp_fin_{pat}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1); Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean())))
    return float(np.mean(Ps)) if len(Ps) == need else float("nan")
# ---- gradient solver: choose learning rate and iteration count on 24 generated images
run = []; g = 0; GR = [(lr, n) for lr in ("0.005", "0.02", "0.05", "0.15") for n in ("100", "400")]
for t, c in T.items():
    cc = c.split()
    for lr, n in GR:
        while len([p for p in run if p.poll() is None]) >= 16: time.sleep(3)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % 8), DATA=os.path.realpath(f"{L}/data_gen512"), SHARD="0:24", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", OMP_NUM_THREADS="4", SOLVER="repo", REPO_LR=lr, REPO_N=n)
        env.update(dict(kv.split("=", 1) for kv in cc[1:])); g += 1
        run.append(subprocess.Popen([P, f"{L}/unified.py", f"fin_attg_{t}_{lr}_{n}", f"--deg={cc[0]}"], env=env, stdout=open(f"{L}/logs_x/attg_{t}_{lr}_{n}.out", "w"), stderr=subprocess.STDOUT))
[p.wait() for p in run]; best = {}
for t in T:
    R = {(lr, n): psnr_dir(f"attg_{t}_{lr}_{n}", 24) for lr, n in GR}; ok = {k: v for k, v in R.items() if v == v}
    best[t] = max(ok, key=ok.get) if ok else ("0.05", "100")
    print(f"GRADSEL {t}: " + " | ".join(f"lr {lr} x{n}: {v:.2f}" for (lr, n), v in R.items()) + f" || chosen lr {best[t][0]}, {best[t][1]} iterations", flush=True)
# ---- all arms through the same fit-and-test launcher
ARMS = [("ours", t, c) for t, c in T.items()] + [("grad", t, f"{c} SOLVER=repo REPO_LR={best[t][0]} REPO_N={best[t][1]}") for t, c in T.items()] + \
       [(a, t, f"{T[t]} GENERIC={'1' if a == 'generic' else 'mean'}") for a in ("generic", "genmean") for t in ("hdr", "quant", "onebit", "gamma")]
run = []
for i, (a, t, c) in enumerate(ARMS):
    while len([p for p in run if p.poll() is None]) >= int(os.environ.get("MAXL", "5")): time.sleep(5)
    run.append(subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=f"att{a}{t}", KEY=t, CFGSTR=c, NSTEP="4", MODES="zband8", CALSH="0:64 64:128", NGPU="8", PAR="4", GOFF=str(2 * i)), stdout=open(f"{L}/logs_x/launch_att{a}{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L))
[p.wait() for p in run]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def score(a, t):
    Ps = []; Ls = []; Tm = []
    for d in glob.glob(f"{L}/exp_fin_att{a}{t}test_zband8_{t}_*_s*"):
        if os.path.exists(d + "/stats.json"): Tm += [s.get("time_total", np.nan) for s in json.load(open(d + "/stats.json"))[2:]]
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
            with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return f"{np.mean(Ps):.2f}/{np.mean(Ls):.3f}, {np.nanmedian(Tm):.2f}s" + ("" if len(Ps) == 107 else f" (n={len(Ps)})") if Ps else "-"
print("| Task | Ours (closed form) | Gradient solver, tuned | Generic table, minimiser | Generic table, mean |\n|---|---|---|---|---|", flush=True)
for t in T: print(f"| {t} | {score('ours', t)} | {score('grad', t)} (lr {best[t][0]}, {best[t][1]} it) | {score('generic', t)} | {score('genmean', t)} |", flush=True)
