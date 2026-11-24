"""ATTEMPT 2 (run from PnP-CM/), everything tuned on generated images:
A. NON-ITERATIVE unmasked phase retrieval + the shared components (noise tilt, tuned schedule, 8-band + data-fit combination).
B. MASKED phase retrieval kept in the operator class: A(x) = |F P D x| solved by the two-piece data step (PR2=1) instead of the reduction to inpainting
   (attempt 1).  Coupling weight (3 values) and schedule chosen on generated images.   Test: 107 images (A: x 3 measurement-noise draws)."""
import json, subprocess, os, glob, re, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; gs = lambda a, n: [float(v) for v in np.geomspace(a, 0.03, n)]
ND = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_NET=v2k8 PRD_INIT=netdirect PRD_TV=0 PRD_NETR=32"
PRM = "INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=prior LINEXACT=1 PR2=1"; LIN = [0.327, 0.132, 0.01, 0.01]
K = {"prd_t0": ["deblur_nl", f"{ND} TILT=0", gs(0.7, 4)], "prd_t05": ["deblur_nl", f"{ND} TILT=0.5", gs(0.7, 4)],
     **{f"prmask__pr2_{r}": ["phase_retrieval", f"{PRM} PR2_RS={r}", LIN] for r in ("0.3", "1", "3")}}
ex = json.load(open(f"{L}/sched_cfg_extra.json")) if os.path.exists(f"{L}/sched_cfg_extra.json") else {}; ex.update(K); json.dump(ex, open(f"{L}/sched_cfg_extra.json", "w"))
sp = [subprocess.Popen([P, f"{L}/sched_cd2.py", k, str(2 * g)], env=dict(os.environ, NSTEPS="4"), stdout=open(f"{L}/logs_x/att2_cd2_{k}.log", "w"), stderr=subprocess.STDOUT) for g, k in enumerate(K)]
[p.wait() for p in sp]; R = {k: json.load(open(f"{L}/sched_cd2_{k}.json"))["4"] for k in K if os.path.exists(f"{L}/sched_cd2_{k}.json")}
print("A2 tuning on generated images: " + " | ".join(f"{k}: {v['start'][0]:.2f} -> {v['best']:.2f}" for k, v in R.items()), flush=True)
bd = max([k for k in R if k.startswith("prd")], key=lambda k: R[k]["best"]); bm = max([k for k in R if k.startswith("prmask")], key=lambda k: R[k]["best"]); tilt = "0" if bd == "prd_t0" else "0.5"
sg = lambda r: f"SIGS={','.join(f'{v:.4f}' for v in r['sig'])} SIGPOS=0,1,2,3 SIGTOLD={r['told']}"
print(f"A2 chosen: non-iterative PR tilt {tilt}, sigma {' '.join(f'{v:.3f}' for v in R[bd]['sig'])} x{R[bd]['told']} | masked PR {bm}, sigma {' '.join(f'{v:.3f}' for v in R[bm]['sig'])} x{R[bm]['told']}", flush=True)
JOBS = [("a2D4", "prdD4", f"deblur_nl {ND} TILT={tilt} {sg(R[bd])}", "4", "0:107"), ("a2D8", "prdD8", f"deblur_nl {ND} TILT={tilt} SIGS={','.join(f'{v:.4f}' for v in gs(1.0, 8))}", "8", "0:107"),
        ("a2M", "prmask", f"phase_retrieval {K[bm][1]} {sg(R[bm])}", "4", "0:54 54:107")]
run = [subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=t, KEY=k, CFGSTR=c, NSTEP=n, MODES="zband8", CALSH="0:64 64:128", TESTSH=ts, NGPU="8", PAR="2", GOFF=str(2 * i)), stdout=open(f"{L}/logs_x/launch_{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L) for i, (t, k, c, n, ts) in enumerate(JOBS)]
[p.wait() for p in run]; run = []
for i, (t, k, c, n, ts) in enumerate(JOBS[:2]):
    cc = c.split()
    for ms in (1, 2):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str((2 * i + ms) % 8), DATA=os.path.realpath(f"{L}/data_paper"), SHARD="0:107", SAMPLE_SEED="1", MEAS_SEED=str(ms), RESTART="never", NOISE="white", NOLOG="1", MODE="unified", OMP_NUM_THREADS="4", LLE=os.path.realpath(f"{L}/lle/{t}_zband8_{k}.pt"))
        env.update(dict(kv.split("=", 1) for kv in cc[1:])); run.append(subprocess.Popen([P, f"{L}/unified.py", f"fin_{t}m{ms}_zband8_{k}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{t}m{ms}.out", "w"), stderr=subprocess.STDOUT))
[p.wait() for p in run]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def score(dirs):
    Ps = []; Ls = []
    for d in dirs:
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1; x = torch.load(rf)["x_raw"].float().clamp(-1, 1)
            with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x.cuda(), gt.cuda())))
    Ps = np.array(Ps); return f"{Ps.mean():.2f}/{np.mean(Ls):.3f} (n={len(Ps)}), below 14 dB {100 * (Ps < 14).mean():.1f}%" if len(Ps) else "missing"
for t, k, n, prev in (("a2D4", "prdD4", 4, "22.87/0.385"), ("a2D8", "prdD8", 8, "23.72/0.345")):
    print(f"A2 TEST non-iterative unmasked PR + shared components, {n} calls: {score(glob.glob(f'{L}/exp_fin_{t}test_zband8_{k}_deblur_nl_s0') + glob.glob(f'{L}/exp_fin_{t}m[12]_zband8_{k}_s0'))} | without them {prev}", flush=True)
print(f"A2 TEST masked PR, attempt 2 (two-piece data step in the operator class), as produced: {score(glob.glob(f'{L}/exp_fin_a2Mtest_zband8_prmask_phase_retrieval_s*'))} | attempt 1 (reduction to inpainting) 28.20/0.273", flush=True)
