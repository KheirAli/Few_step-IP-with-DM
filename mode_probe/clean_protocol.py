"""CLEAN PROTOCOL (run from PnP-CM/): every per-task choice re-made on MODEL-GENERATED images only, with one rule for all tasks.
For each task: (1) a short list of data-fit variants (the ones compared earlier on real selection images); (2) for each variant, one pass of
coordinate search over the 4 noise levels and the told-level factor (sched_cd2.py, 19 runs, combination in the score, 48 generated images);
(3) the variant + schedule with the best score is adopted - no look at the test set; (4) in-loop combination weights fitted on 128 generated
images; (5) the 107 test images, once."""
import json, subprocess, os, glob, re, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"
LIN = [0.327, 0.132, 0.01, 0.01]; SR = [0.465, 0.245, 0.01, 0.01]; JP = [0.694, 0.571, 0.351, 0.01]; NL = [0.694, 0.635, 0.56, 0.459]; CTB = [0.5, 0.25, 0.1, 0.02]
FW = "FW=1 FWP=0.5"; PRM = "INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=prior LINEXACT=1"
CAND = {"sr_bicubic": [("plain", "sr_bicubic", "LINEXACT=1", SR), ("fw", "sr_bicubic", f"LINEXACT=1 {FW}", SR)],
        "deblur_gauss": [("plain", "deblur_gauss", "LINEXACT=1", LIN), ("fw", "deblur_gauss", f"LINEXACT=1 {FW}", LIN)],
        "inpainting": [("plain", "inpainting", "LINEXACT=1", LIN), ("fw", "inpainting", f"LINEXACT=1 {FW}", LIN)],
        "jpeg": [("soft", "jpeg", "JPROJ=1 JSOFT=1 JCOLD=1", JP), ("softfw", "jpeg", f"JPROJ=1 JSOFT=1 JCOLD=1 {FW} FWSTEPS=1,2,3", JP), ("true", "jpeg", "JNONE=1", JP)],
        "deblur_nl": [("plain", "deblur_nl", "WARMX=lin NLKER=1 MEC=step", NL), ("fw", "deblur_nl", f"WARMX=lin NLKER=1 MEC=step {FW} FWMIN=0.4", NL)],
        "hdr": [("plain", "deblur_nl", "NEWOP=hdr HDRX=1", NL), ("rx2", "deblur_nl", "NEWOP=hdr HDRX=1 RHOX=2", NL)],
        "quant": [("plain", "deblur_nl", "NEWOP=quant", NL), ("fw", "deblur_nl", f"NEWOP=quant {FW} FWMIN=0.4", NL)],
        "onebit": [("nearest", "deblur_nl", "NEWOP=onebit", NL), ("ev", "deblur_nl", "NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2", NL), ("evfw", "deblur_nl", f"NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2 {FW} FWMIN=0.4", NL)],
        "prmask": [("plain", "phase_retrieval", PRM, LIN), ("fw", "phase_retrieval", f"{PRM} {FW}", LIN)],
        "blurclip": [("split", "deblur_nl", "NEWOP=blurclip", NL), ("ev", "deblur_nl", "NEWOP=blurclip SPLIT_EV=0.3", NL)],
        "blurq": [("split", "deblur_nl", "NEWOP=blurq", NL)], "gamma": [("plain", "deblur_nl", "NEWOP=gamma", NL)],
        "ct30": [("r1", "deblur_nl", "NEWOP=ct CT_K=30 CT_SVD=1 RHOX=1", CTB), ("r3", "deblur_nl", "NEWOP=ct CT_K=30 CT_SVD=1 RHOX=3", CTB)],
        "ct60": [("r1", "deblur_nl", "NEWOP=ct CT_K=60 CT_SVD=1 RHOX=1", CTB), ("r3", "deblur_nl", "NEWOP=ct CT_K=60 CT_SVD=1 RHOX=3", CTB)]}
PREV = {"sr_bicubic": "28.28/0.297", "deblur_gauss": "29.76/0.263", "inpainting": "30.29/0.207", "jpeg": "27.02/0.338", "deblur_nl": "26.43/0.332", "hdr": "26.42/0.268", "quant": "31.46/0.177", "onebit": "30.30/0.213",
        "prmask": "27.93/0.271", "blurclip": "23.62/0.433 (no combination)", "blurq": "25.52/0.413 (no combination)", "gamma": "29.21/0.217 (no combination)", "ct30": "29.20/0.255 (no combination)", "ct60": "31.50/0.200 (no combination)"}
# env: PREFIX (name space for another dataset), TASKS (subset), NTEST (107), DSET / GENDATA / CALDATA / TESTDATA / TESTSH (other dataset; inherited by the runs)
PRE = os.environ.get("PREFIX", ""); NTEST = int(os.environ.get("NTEST", "107"))
CAND.update({f"mri{R}": [(f"r{r}", "deblur_nl", f"NEWOP=mri MRI_R={R} RHOX={r}", CTB) for r in ("1", "3")] for R in (4, 8)})
if os.environ.get("TASKS"): CAND = {t: CAND[t] for t in os.environ["TASKS"].split()}
else: CAND = {t: c for t, c in CAND.items() if not t.startswith("mri")}
keys = {f"{PRE}{t}__{n}": (deg, ex, base) for t, cs in CAND.items() for n, deg, ex, base in cs}
ex0 = json.load(open(f"{L}/sched_cfg_extra.json")) if os.path.exists(f"{L}/sched_cfg_extra.json") else {}; ex0.update({k: list(v) for k, v in keys.items()}); json.dump(ex0, open(f"{L}/sched_cfg_extra.json", "w"))
run = []; WAVE = int(os.environ.get("WAVE", "12"))
for g, k in enumerate(keys):
    while len([p for p in run if p.poll() is None]) >= WAVE: time.sleep(5)
    run.append(subprocess.Popen([P, f"{L}/sched_cd2.py", k, str(g)], env=dict(os.environ, NSTEPS="4"), stdout=open(f"{L}/logs_x/clean_cd2_{k}.log", "w"), stderr=subprocess.STDOUT))
[p.wait() for p in run]; chosen = {}
for t, cs in CAND.items():
    R = {n: json.load(open(f"{L}/sched_cd2_{PRE}{t}__{n}.json"))["4"] for n, *_ in cs if os.path.exists(f"{L}/sched_cd2_{PRE}{t}__{n}.json")}
    R = {n: v for n, v in R.items() if v["best"] == v["best"]}
    if not R: print(f"CHOICE {t}: no valid candidate", flush=True); continue
    b = max(R, key=lambda n: R[n]["best"]); chosen[t] = (b, R[b])
    print(f"CHOICE {t}: " + " | ".join(f"{n}: {v['start'][0]:.2f}->{v['best']:.2f}" for n, v in R.items()) + f" || chosen {b}, sigma {' '.join(f'{x:.3f}' for x in R[b]['sig'])}, told x{R[b]['told']}", flush=True)
json.dump({t: {"variant": b, **r} for t, (b, r) in chosen.items()}, open(f"{L}/clean_protocol_choices{PRE}.json", "w"))
run = []
for i, (t, (b, r)) in enumerate(chosen.items()):
    while len([p for p in run if p.poll() is None]) >= 7: time.sleep(5)
    deg, ex, _ = keys[f"{PRE}{t}__{b}"]; cfg = f"{deg} {ex} SIGS={','.join(f'{v:.4f}' for v in r['sig'])} SIGPOS=0,1,2,3 SIGTOLD={r['told']}"
    run.append(subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=f"cln{PRE}{t}", KEY=t, CFGSTR=cfg, NSTEP="4", MODES="zband8", CALSH="0:64 64:128", NGPU="8", PAR="4", GOFF=str(2 * i)), stdout=open(f"{L}/logs_x/launch_cln{PRE}{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L))
[p.wait() for p in run]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
print("| Task | Chosen on generated images | Clean protocol, test | Previous number |\n|---|---|---|---|", flush=True)
for t, (b, r) in chosen.items():
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_cln{PRE}{t}test_zband8_{t}_*_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    print(f"| {t} | {b}; sigma {' '.join(f'{x:.2f}' for x in r['sig'])}; told x{r['told']} | " + (f"{np.mean(Ps):.2f}/{np.mean(Ls):.3f}" + ("" if len(Ps) == NTEST else f" (n={len(Ps)})") if Ps else "missing") + f" | {PREV.get(t, '-') if not PRE else '-'} |", flush=True)
