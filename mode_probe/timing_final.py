"""Wall clock of the FINAL configuration of every table row (run from PnP-CM/): one image at a time on one GPU, 12 test images, median of the last 10.
'total' = everything per image (start stage, data steps, model calls, combination); 'sampling' = the loop without the start stage."""
import json, subprocess, os, glob, sys, numpy as np
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; exec(open(f"{L}/final_cfg.py").read()); G = os.environ.get("TGPU", "6")
print("| Row | Total s / image | Loop only s | Combination weights found |\n|---|---|---|---|", flush=True)
for k, (name, (deg, ex, w, n, key)) in enumerate(FINAL.items()):
    if sys.argv[1:] and name not in sys.argv[1:]: continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=G, DATA=os.path.realpath(f"{L}/data_paper"), SHARD="0:12", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", OMP_NUM_THREADS="4")
    env.update(dict(kv.split("=", 1) for kv in ex.split())); wf = f"{L}/lle/{w}"; env["LLE"] = os.path.realpath(wf) if os.path.exists(wf) else "/nonexistent.json"
    subprocess.run([P, f"{L}/unified.py", f"fin_tf_{k}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/tf_{name}.out", "w"), stderr=subprocess.STDOUT)
    f = glob.glob(f"{L}/exp_fin_tf_{k}_s0/stats.json"); S = json.load(open(f[0]))[2:] if f else []
    print(f"| {name} | {np.median([s['time_total'] for s in S]):.2f} | {np.median([s['time'] for s in S]):.2f} | {'yes' if os.path.exists(wf) else 'NO'} |" if S else f"| {name} | failed | | |", flush=True); subprocess.run(f"rm -rf {L}/exp_fin_tf_{k}_s0", shell=True)
if os.environ.get("TREF"):      # PnP-CM references measured in the same session on the same GPU (same loop, its own solver, no tilt, no combination)
    REF = {"PnP-CM sr_bicubic (exact linear step)": ("sr_bicubic", "LINEXACT=1 TILT=0"), "PnP-CM hdr": ("deblur_nl", "NEWOP=hdr SOLVER=repo TILT=0"), "PnP-CM jpeg": ("jpeg", "SOLVER=repo TILT=0"), "PnP-CM deblur_nl": ("deblur_nl", "SOLVER=repo TILT=0"),
           "PnP-CM blurq": ("deblur_nl", "NEWOP=blurq SOLVER=repo TILT=0"), "PnP-CM ctnl30": ("deblur_nl", "NEWOP=ctnl CT_K=30 SOLVER=repo RHOX=1 REPO_LR=3e-2 TILT=0"), "PnP-CM cdp2": ("deblur_nl", "NEWOP=prcdp PR_MASKS=2 SOLVER=repo REPO_LR=3e-2 TILT=0"),
           "PnP-CM prmask": ("phase_retrieval", "SOLVER=repo TILT=0"), "PnP-CM mri4 (exact linear step)": ("deblur_nl", "NEWOP=mri MRI_R=4 RHOX=1 TILT=0")}
    for k, (name, (deg, ex)) in enumerate(REF.items()):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=G, DATA=os.path.realpath(f"{L}/data_paper"), SHARD="0:12", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", OMP_NUM_THREADS="4"); env.update(dict(kv.split("=", 1) for kv in ex.split()))
        subprocess.run([P, f"{L}/unified.py", f"fin_tr_{k}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/tr_{k}.out", "w"), stderr=subprocess.STDOUT)
        f = glob.glob(f"{L}/exp_fin_tr_{k}_s0/stats.json"); S = json.load(open(f[0]))[2:] if f else []
        print(f"| {name} | {np.median([s['time_total'] for s in S]):.2f} | {np.median([s['time'] for s in S]):.2f} | - |" if S else f"| {name} | failed | | |", flush=True); subprocess.run(f"rm -rf {L}/exp_fin_tr_{k}_s0", shell=True)
