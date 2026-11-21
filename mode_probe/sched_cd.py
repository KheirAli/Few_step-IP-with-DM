"""Cheap schedule tuning: ONE pass of coordinate search. For step 1..4 in turn, try 4 other values of that step's noise level
(x0.3, x0.55, x1.8, x3.3; steps at ~0 use a floor of 0.05) with the rest fixed, keep the best; then the told-level factor (1.0, 1.5).
18 runs + 1 baseline per task on 48 model-generated images; checked on the 24 real selection images.   usage: python ../mode_probe/sched_cd.py <task> <gpu offset>"""
import sys, os, glob, re, json, subprocess, shutil, numpy as np, torch
from PIL import Image
sys.argv = [sys.argv[0], sys.argv[1], "48"] + sys.argv[2:]; task = sys.argv[1]; OFF = int(sys.argv[3]) if len(sys.argv) > 3 else 0; L = "../mode_probe"
src = open(f"{L}/sched_search.py").read(); exec(src[src.index("CFG = {"):src.index("def psnr_of")]); exec(src[src.index("def psnr_of"):src.index("def evaluate")])
deg, extra, base = CFG[task]; DATA = os.path.realpath(f"{L}/data_gen512")
def run(cands, name, data=DATA, n=48):
    pr = []
    for k, (sig, told) in enumerate(cands):
        tag = f"cd_{task}_{name}_{k}"; env = dict(os.environ, CUDA_VISIBLE_DEVICES=str((OFF + k) % 8), DATA=data, SHARD=f"0:{n}", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5",
                                                   SIGS=",".join(f"{v:.4f}" for v in sig), SIGPOS="0,1,2,3", SIGTOLD=f"{told:.3f}", OMP_NUM_THREADS="4"); env.update(dict(kv.split("=", 1) for kv in extra.split()))
        pr.append((tag, subprocess.Popen(["../pnpcm_env/bin/python", f"{L}/unified.py", f"fin_{tag}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)))
    out = []
    for tag, p in pr: p.wait(); out.append(psnr_of(tag, n)); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
    return out
cur = (list(base), 1.2); best = run([cur], "b")[0]; b0 = best; nruns = 1
for k in range(4):
    cs = []
    for m in (0.3, 0.55, 1.8, 3.3):
        s = list(cur[0]); s[k] = max(s[k], 0.05) * m; cs.append((s, cur[1]))
    r = run(cs, f"s{k}"); nruns += 4; j = int(np.nanargmax(r))
    if r[j] > best: best = r[j]; cur = cs[j]
cs = [(cur[0], 1.0), (cur[0], 1.5)]; r = run(cs, "t"); nruns += 2; j = int(np.nanargmax(r))
if r[j] > best: best = r[j]; cur = cs[j]
chk = run([(list(base), 1.2), cur], "chk", data=os.path.realpath(f"{L}/data100"), n=24); full = json.load(open(f"{L}/sched_search_{task}.json"))
print(f"CD {task:12s} {nruns} runs: generated {b0:.2f} -> {best:.2f} dB (full search {full['top'][0][0]:.2f}) | sigma " + " ".join(f"{v:.3f}" for v in cur[0]) + f" told x{cur[1]} | 24 real selection images: current {chk[0]:.2f} -> {chk[1]:.2f} dB ({chk[1] - chk[0]:+.2f})", flush=True)
json.dump({"sig": cur[0], "told": cur[1]}, open(f"{L}/sched_cd_{task}.json", "w"))
