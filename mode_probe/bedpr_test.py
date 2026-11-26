"""Bedroom unmasked phase retrieval, TEST: the 58 test images (data_bed), 3 measurement-noise draws (MEAS_SEED 0, 1, 2), with the band-combination
weights fitted on generated images (lle/<tag>_zband8_<key>.pt).  A run is repeated only if it died (out of memory).  usage (from PnP-CM/): bedpr_test.py <tag> [tag ...]"""
import sys, os, json, glob, subprocess, shutil, time
MP = "../mode_probe/bedpr_maxp_test.txt"
MAXP = int(os.environ.get("MAXP", "3")); open(MP, "w").write(str(MAXP)) if not os.path.exists(MP) else None; exec(open("../mode_probe/bedpr_cfg.py").read())
def launch(t, k, c, ms):
    tag = f"{t}m{ms}_zband8_{k}"; g = subprocess.run(["bash", f"{L}/bedpr_pick.sh", "8000"], capture_output=True, text=True).stdout.strip(); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
    lle = os.path.realpath(f"{L}/lle/{t}_zband8_{k}.pt"); assert os.path.exists(lle), lle
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, DATA=os.path.realpath(f"{L}/data_bed"), SHARD="0:58", SAMPLE_SEED="1", MEAS_SEED=str(ms), RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", OMP_NUM_THREADS="4", LLE=lle, **BENV)
    env.update(dict(kv.split("=", 1) for kv in c.split()[1:])); env.update(W0(t[5:], "test"))
    return subprocess.Popen(["../pnpcm_env/bin/python", f"{L}/unified.py", f"fin_{tag}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)
todo = [(t, k, c, ms, 0) for t, k, c, n, d in JOBS if t in sys.argv[1:] for ms in (0, 1, 2) if len(glob.glob(f"{L}/exp_fin_{t}m{ms}_zband8_{k}_s0/raw_*.pt")) != 58]; act = []
while todo or act:
    for it in list(act):
        t, k, c, ms, tr, p = it
        if p.poll() is None: continue
        act.remove(it); n = len(glob.glob(f"{L}/exp_fin_{t}m{ms}_zband8_{k}_s0/raw_*.pt")); print(f"BEDTEST {t} draw {ms}: {n} images (try {tr + 1})", flush=True)
        if n != 58 and tr < 8: todo.append((t, k, c, ms, tr + 1))
    if todo and len(act) < int(open(MP).read()): t, k, c, ms, tr = todo.pop(0); act.append((t, k, c, ms, tr, launch(t, k, c, ms))); time.sleep(40); continue
    time.sleep(5)
print("BEDTEST ALLDONE", flush=True)
