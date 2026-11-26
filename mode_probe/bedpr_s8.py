"""First noise level of the 8-call geometric schedule for bedroom unmasked phase retrieval, chosen as pr_clean.py stage 2 did for faces:
candidates 0.75, 1.0, 1.5, 2.0, 3.0, plain PSNR of the sampler output on 24 MODEL-GENERATED bedroom images (data_bedgen 0:24), tilt 0.5.
usage (from PnP-CM/): bedpr_s8.py <free|direct>   -> adds the choice to ../mode_probe/bedpr_s8.json"""
import sys, os, json, glob, subprocess, shutil, time, numpy as np
task = sys.argv[1]; MAXP = int(os.environ.get("MAXP", "2")); exec(open("../mode_probe/bedpr_cfg.py").read()); cfg = {"free": FREE, "direct": ND}[task]; CAND = (0.75, 1.0, 1.5, 2.0, 3.0)
def launch(tag, a):
    g = subprocess.run(["bash", f"{L}/bedpr_pick.sh", "8000"], capture_output=True, text=True).stdout.strip(); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, DATA=os.path.realpath(f"{L}/data_bedgen"), SHARD="0:24", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", OMP_NUM_THREADS="4", LLE="/nonexistent_weights.json", **BENV)
    env.update(dict(kv.split("=", 1) for kv in cfg.split())); env["SIGS"] = ",".join(f"{v:.4f}" for v in gs(a, 8))
    return subprocess.Popen(["../pnpcm_env/bin/python", f"{L}/unified.py", f"fin_{tag}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)
todo = [(f"bedprs8_{task}_{a}", a, 0) for a in CAND]; act = []; res = {}
while todo or act:
    for it in list(act):
        tag, a, tr, p = it
        if p.poll() is None: continue
        act.remove(it); f = f"{L}/exp_fin_{tag}_s0/stats.json"; ps = [s["psnr"] for s in json.load(open(f))] if os.path.exists(f) else []
        if len(ps) == 24 or tr >= 6: res[a] = float(np.mean(ps)) if len(ps) == 24 else float("nan"); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
        else: todo.append((tag, a, tr + 1))
    if todo and len(act) < MAXP: tag, a, tr = todo.pop(0); act.append((tag, a, tr, launch(tag, a))); time.sleep(40); continue
    time.sleep(5)
b = max(res, key=lambda a: res[a] if res[a] == res[a] else -1); print(f"BEDS8 {task}: " + " | ".join(f"{a}: {res[a]:.2f}" for a in CAND) + f" || chosen {b}", flush=True)
f = f"{L}/bedpr_s8.json"; d = json.load(open(f)) if os.path.exists(f) else {}; d[task] = b; d[task + "_scores"] = {str(a): res[a] for a in CAND}; json.dump(d, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
