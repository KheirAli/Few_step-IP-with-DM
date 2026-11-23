"""Three statistical ideas, each tuned on 48 MODEL-GENERATED images and then run once on the 107 test images (run from PnP-CM/).
A  expected-value per-ray update, nonlinear CT (30 views)      B  expected-value per-pixel update, blur then clip
C  regression start for the clipped region, HDR                D  Poisson CT: Lambert-W per-ray update vs logarithm of the counts"""
import subprocess, os, glob, re, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; MAXP = 16
CT = "NEWOP=ctnl CT_K=30 SIGS=0.5,0.25,0.1,0.02 SIGPOS=0,1,2,3"
G = {"A": {f"ev{t}": f"{CT} RHOX=0.3" + (f" SPLIT_EV={t}" if t != "0" else "") for t in ("0", "0.3", "1", "3")},
     "B": {f"ev{t}": "NEWOP=blurclip" + (f" SPLIT_EV={t}" if t != "0" else "") for t in ("0", "0.3", "1", "3", "10")},
     "C": {f"{s}_{r}": "NEWOP=hdr HDRX=1" + (" SIGS=3.0,0.635,0.56,0.459 SIGPOS=0,1,2,3" if s == "s3" else "") + ("" if r == "off" else f" HDR_REG=1 HDR_REGK={r}") for s in ("s0", "s3") for r in ("off", "64", "256")},
     "D": {**{f"lw{r}": f"{CT} CTNL_MODE=poisson RHOX={r}" for r in ("0.1", "0.3", "1", "3")}, **{f"log{r}": f"{CT} CTNL_MODE=poislog RHOX={r}" for r in ("3", "10", "30", "100")}}}
def launch(tag, cfg, data, shard, g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % 8), DATA=os.path.realpath(data), SHARD=shard, SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", OMP_NUM_THREADS="4")
    env.update(dict(kv.split("=", 1) for kv in cfg.split()))
    return subprocess.Popen([P, f"{L}/unified.py", f"fin_{tag}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{tag}_{shard.split(':')[0]}.out", "w"), stderr=subprocess.STDOUT)
def run_all(jobs):
    run = []; g = 0
    for j in jobs:
        while len([p for p in run if p.poll() is None]) >= MAXP: time.sleep(2)
        run.append(launch(*j, g)); g += 1
    [p.wait() for p in run]
lp = None
def score(tag, with_lpips=False):
    global lp; Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{tag}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean())))
        if with_lpips:
            if lp is None: lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
            with torch.no_grad(): Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return (float(np.mean(Ps)) if Ps else float("nan"), float(np.mean(Ls)) if Ls else float("nan"), len(Ps))
run_all([(f"i3g_{k}_{n}", c, f"{L}/data_gen512", "0:48") for k in G for n, c in G[k].items()])
sel = {}
for k in G:
    R = {n: score(f"i3g_{k}_{n}") for n in G[k]}; print(f"GEN {k}: " + " | ".join(f"{n} {v[0]:.2f}" + ("" if v[2] == 48 else f" (n={v[2]})") for n, v in R.items()), flush=True)
    ok = {n: v[0] for n, v in R.items() if v[2] == 48 and v[0] == v[0]}
    if k == "D": sel[k] = [max([n for n in ok if n.startswith("lw")], key=ok.get), max([n for n in ok if n.startswith("log")], key=ok.get)]
    elif k == "C": sel[k] = ["s3_off", max([n for n in ok if n.startswith("s3") and not n.endswith("off")], key=ok.get), "s0_off", max([n for n in ok if n.startswith("s0") and not n.endswith("off")], key=ok.get)]
    else: sel[k] = ["ev0", max([n for n in ok if n != "ev0"], key=ok.get)]
run_all([(f"i3t_{k}_{n}", G[k][n], f"{L}/data_paper", sh) for k in sel for n in sel[k] for sh in ("0:54", "54:107")])
for k in sel:
    print(f"TEST {k}: " + " | ".join((lambda v: f"{n} {v[0]:.2f}/{v[1]:.3f} (n={v[2]})")(score(f"i3t_{k}_{n}", True)) for n in sel[k]), flush=True)
