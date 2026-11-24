"""Clean timing pass (run from PnP-CM/, when the GPUs are otherwise idle): seconds per image for each task's clean-protocol configuration with its
fitted combination weights, one image at a time on one GPU, 12 test images (the first two are discarded as warm-up)."""
import json, subprocess, os, glob, numpy as np
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; ch = json.load(open(f"{L}/clean_protocol_choices.json"))
src = open(f"{L}/clean_protocol.py").read(); exec(src[src.index("LIN = ["):src.index("PREV = {")])
J = []
for t, c in ch.items():
    deg, ex = [(d, e) for n, d, e, b in CAND[t] if n == c["variant"]][0]; J.append((t, deg, f"{ex} SIGS={','.join(f'{v:.4f}' for v in c['sig'])} SIGPOS=0,1,2,3 SIGTOLD={c['told']}", f"{L}/lle/cln{t}_zband8_{t}.pt"))
J += [("ctnl30 (no alternation)", "deblur_nl", "NEWOP=ctnl CT_K=30 RHOX=0.03 CTNL_N=1 CTNL_MU=1 SIGS=0.9,0.45,0.594,0.165 SIGPOS=0,1,2,3 SIGTOLD=1.5", f"{L}/lle/ctnlO30_zband8_ctnl30.pt"),
      ("ctnl60 (no alternation)", "deblur_nl", "NEWOP=ctnl CT_K=60 RHOX=0.03 CTNL_N=1 CTNL_MU=1 SIGS=0.9,0.45,0.594,0.165 SIGPOS=0,1,2,3 SIGTOLD=1.5", f"{L}/lle/ctnlO60_zband8_ctnl60.pt"),
      ("unmasked PR, non-iterative, 4 calls", "deblur_nl", "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_NET=v2k8 PRD_INIT=netdirect PRD_TV=0 PRD_NETR=32 TILT=0 SIGS=0.7000,0.2450,0.0857,0.0300", ""),
      ("unmasked PR, iterative training-free, 4 calls", "deblur_nl", "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_INIT=regfit4 PRD_NSTART=16 PRD_TV=80 TILT=0 SIGS=1.0000,0.3107,0.0965,0.0300", ""),
      ("PnP-CM gradient solver, HDR", "deblur_nl", "NEWOP=hdr SOLVER=repo TILT=0", ""), ("PnP-CM gradient solver, JPEG", "jpeg", "SOLVER=repo TILT=0", ""), ("PnP-CM gradient solver, nonlinear deblur", "deblur_nl", "SOLVER=repo TILT=0", ""), ("PnP-CM, SR (exact linear step)", "sr_bicubic", "LINEXACT=1 TILT=0", "")]
print("| Configuration | Seconds per image (median of 10) |\n|---|---|", flush=True)
for k, (t, deg, ex, w) in enumerate(J):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("TGPU", "0"), DATA=os.path.realpath(f"{L}/data_paper"), SHARD="0:12", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", OMP_NUM_THREADS="4")
    env.update(dict(kv.split("=", 1) for kv in ex.split()))
    if w and os.path.exists(w): env["LLE"] = os.path.realpath(w)
    subprocess.run([P, f"{L}/unified.py", f"fin_tp_{k}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/tp_{k}.out", "w"), stderr=subprocess.STDOUT)
    f = glob.glob(f"{L}/exp_fin_tp_{k}_s0/stats.json"); tm = [s.get("time_total", s.get("time", np.nan)) for s in json.load(open(f[0]))[2:]] if f else []
    print(f"| {t}" + ("" if (not w or os.path.exists(w)) else " (weights missing)") + f" | {np.nanmedian(tm):.2f} |" if tm else f"| {t} | failed |", flush=True); subprocess.run(f"rm -rf {L}/exp_fin_tp_{k}_s0", shell=True)
