"""Unmasked phase retrieval with TODAY'S components on the model steps (run from PnP-CM/), all tuned on generated images:
noise tilt (0 vs 0.5), noise schedule by one pass of coordinate search with the combination in the score, and the in-loop combination
(8 frequency bands + data-fit term).  Starting stage as chosen in the clean protocol (80 classical passes, offset on; predictor radius 48).
Test: 107 images x 3 measurement-noise draws."""
import json, subprocess, os, glob, re, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; gs = lambda a, n: [float(v) for v in np.geomspace(a, 0.03, n)]
B = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_TV=80"; FREE = f"{B} PRD_INIT=regfit4 PRD_NSTART=16"; NET = f"{B} PRD_INIT=netfit PRD_NET=v2k8 PRD_NETREG=0 PRD_NETR=48"
ex = json.load(open(f"{L}/sched_cfg_extra.json")) if os.path.exists(f"{L}/sched_cfg_extra.json") else {}
ex.update({"pru_t0": ["deblur_nl", f"{FREE} TILT=0", gs(1.0, 4)], "pru_t05": ["deblur_nl", f"{FREE} TILT=0.5", gs(1.0, 4)]}); json.dump(ex, open(f"{L}/sched_cfg_extra.json", "w"))
sp = [subprocess.Popen([P, f"{L}/sched_cd2.py", k, str(g)], env=dict(os.environ, NSTEPS="4"), stdout=open(f"{L}/logs_x/prtoday_cd2_{k}.log", "w"), stderr=subprocess.STDOUT) for g, k in ((0, "pru_t0"), (4, "pru_t05"))]
[p.wait() for p in sp]; R = {k: json.load(open(f"{L}/sched_cd2_{k}.json"))["4"] for k in ("pru_t0", "pru_t05") if os.path.exists(f"{L}/sched_cd2_{k}.json")}
print("PRT tuning on generated images (4 calls, with final combination): " + " | ".join(f"{k}: start {v['start'][0]:.2f} (plain last output {v['start'][1]:.2f}) -> tuned {v['best']:.2f}" for k, v in R.items()), flush=True)
bk = max(R, key=lambda k: R[k]["best"]); tilt = "0" if bk == "pru_t0" else "0.5"; r = R[bk]
print(f"PRT chosen: tilt {tilt}, sigma {' '.join(f'{v:.3f}' for v in r['sig'])}, told x{r['told']}", flush=True)
S4 = f"SIGS={','.join(f'{v:.4f}' for v in r['sig'])} SIGPOS=0,1,2,3 SIGTOLD={r['told']} TILT={tilt}"; S8 = f"SIGS={','.join(f'{v:.4f}' for v in gs(1.5, 8))} TILT={tilt}"
JOBS = [("prtF4", "pruF4", f"deblur_nl {FREE} {S4}", "4"), ("prtN4", "pruN4", f"deblur_nl {NET} {S4}", "4"), ("prtF8", "pruF8", f"deblur_nl {FREE} {S8}", "8"), ("prtN8", "pruN8", f"deblur_nl {NET} {S8}", "8")]
run = [subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=t, KEY=k, CFGSTR=c, NSTEP=n, MODES="zband8", CALSH="0:64 64:128", TESTSH="0:107", NGPU="8", PAR="2", GOFF=str(2 * i)), stdout=open(f"{L}/logs_x/launch_{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L) for i, (t, k, c, n) in enumerate(JOBS)]
[p.wait() for p in run]
# two more measurement-noise draws on the test set with the fitted weights
run = []
for i, (t, k, c, n) in enumerate(JOBS):
    cc = c.split()
    for ms in (1, 2):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str((2 * i + ms) % 8), DATA=os.path.realpath(f"{L}/data_paper"), SHARD="0:107", SAMPLE_SEED="1", MEAS_SEED=str(ms), RESTART="never", NOISE="white", NOLOG="1", MODE="unified", OMP_NUM_THREADS="4", LLE=os.path.realpath(f"{L}/lle/{t}_zband8_{k}.pt"))
        env.update(dict(kv.split("=", 1) for kv in cc[1:])); run.append(subprocess.Popen([P, f"{L}/unified.py", f"fin_{t}m{ms}_zband8_{k}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{t}m{ms}.out", "w"), stderr=subprocess.STDOUT))
[p.wait() for p in run]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
PREV = {"prtF4": "23.62/0.369", "prtN4": "24.07/0.356", "prtF8": "24.38/0.336", "prtN8": "24.78/0.325"}
for t, k, c, n in JOBS:
    Ps = []; Ls = []
    for d in glob.glob(f"{L}/exp_fin_{t}test_zband8_{k}_deblur_nl_s0") + glob.glob(f"{L}/exp_fin_{t}m[12]_zband8_{k}_s0"):
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1; x = torch.load(rf)["x_raw"].float().clamp(-1, 1)
            with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x.cuda(), gt.cuda())))
    Ps = np.array(Ps); print(f"PRT TEST {'training-free' if 'F' in t else 'learned predictor'}, {n} calls: {Ps.mean():.2f}/{np.mean(Ls):.3f} (n={len(Ps)}), below 14 dB {100 * (Ps < 14).mean():.1f}% | before today's components {PREV[t]}" if len(Ps) else f"PRT TEST {t} missing", flush=True)
