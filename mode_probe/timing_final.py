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
