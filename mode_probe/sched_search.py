"""Black-box search of the 4-step noise schedule (and the level told to the model), per task.
Objective: mean PSNR on MODEL-GENERATED images (no real data), same seed for every candidate, no extrapolation.
Random search in log space around the current schedule, then two rounds of local refinement around the best candidates.
usage (from PnP-CM/): python ../mode_probe/sched_search.py <task> [n_images=48]      env: PAR (16), NGPU (8)"""
import sys, os, glob, re, json, subprocess, shutil, numpy as np, torch
from PIL import Image
task = sys.argv[1]; NI = int(sys.argv[2]) if len(sys.argv) > 2 else 48; PAR = int(os.environ.get("PAR", "16")); NGPU = int(os.environ.get("NGPU", "8")); L = "../mode_probe"
CFG = {"sr_bicubic": ("sr_bicubic", "LINEXACT=1", [0.465, 0.245, 0.01, 0.01]), "deblur_gauss": ("deblur_gauss", "LINEXACT=1", [0.327, 0.132, 0.01, 0.01]),
       "inpainting": ("inpainting", "LINEXACT=1 FW=1 FWP=0.5", [0.327, 0.132, 0.01, 0.01]), "jpeg": ("jpeg", "JPROJ=1 JSOFT=1 JCOLD=1 FW=1 FWP=0.5 FWSTEPS=1,2,3", [0.694, 0.571, 0.351, 0.01]),
       "hdr": ("deblur_nl", "NEWOP=hdr HDRX=1", [0.694, 0.635, 0.56, 0.459]), "quant": ("deblur_nl", "NEWOP=quant FW=1 FWP=0.5 FWMIN=0.4", [0.694, 0.635, 0.56, 0.459]),
       "onebit": ("deblur_nl", "NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2 FW=1 FWP=0.5 FWMIN=0.4", [0.694, 0.635, 0.56, 0.459])}
deg, extra, base = CFG[task]; rng = np.random.default_rng(0); DATA = os.path.realpath(f"{L}/data_gen512")
def psnr_of(tag, data_n):
    P = []
    for rf in glob.glob(f"{L}/exp_fin_{tag}_s0/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        P.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean())))
    return float(np.mean(P)) if len(P) == data_n else float("nan")
def evaluate(cands, rnd, data=DATA, n=NI, keep=False):
    out = []
    for s in range(0, len(cands), PAR):
        procs = []
        for k, (sig, told) in enumerate(cands[s:s + PAR]):
            tag = f"ss_{task}_{rnd}_{s + k}"; env = dict(os.environ, CUDA_VISIBLE_DEVICES=str((s + k) % NGPU), DATA=data, SHARD=f"0:{n}", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5",
                                                         SIGS=",".join(f"{v:.4f}" for v in sig), SIGPOS="0,1,2,3", SIGTOLD=f"{told:.3f}", OMP_NUM_THREADS="4")
            env.update(dict(kv.split("=", 1) for kv in extra.split()))
            procs.append((tag, subprocess.Popen(["../pnpcm_env/bin/python", f"{L}/unified.py", f"fin_{tag}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)))
        for (tag, p), c in zip(procs, cands[s:s + PAR]):
            p.wait(); out.append((psnr_of(tag, n), c))
            if not keep: shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
    return out
def jitter(c, sd): return ([float(np.clip(v * np.exp(rng.normal(0, sd)), 0.01, 20.0)) for v in c[0]], float(np.clip(c[1] * np.exp(rng.normal(0, sd / 3)), 1.0, 2.0)))
start = (base, 1.2); ref = [max(v, 0.1) for v in base]
cands = [start] + [([float(r * np.exp(rng.uniform(np.log(0.3), np.log(4.0)))) for r in ref], float(rng.choice([1.0, 1.2, 1.5]))) for _ in range(47)]
hist = evaluate(cands, 0)
for rnd, sd, m in ((1, 0.25, 32), (2, 0.12, 32)):
    top = [c for _, c in sorted([h for h in hist if h[0] == h[0]], key=lambda t: -t[0])[:4]]; hist += evaluate([jitter(top[k % 4], sd) for k in range(m)], rnd)
hist = sorted([h for h in hist if h[0] == h[0]], key=lambda t: -t[0]); b0 = [h for h in hist if h[1] == start][0][0]
print(f"SS {task}: current schedule {base} told x1.2 -> {b0:.2f} dB on {NI} generated images; {len(hist)} candidates evaluated")
for p, (sig, told) in hist[:5]: print(f"SS {task}: {p:.2f} dB ({p - b0:+.2f})  sigma " + " ".join(f"{v:.3f}" for v in sig) + f"  told x{told:.2f}")
json.dump({"task": task, "base": b0, "top": [(p, c) for p, c in hist[:10]]}, open(f"{L}/sched_search_{task}.json", "w"))
# independent check on the 24 real selection images: current schedule against the best found
chk = evaluate([start, hist[0][1]], "chk", data=os.path.realpath(f"{L}/data100"), n=24)
print(f"SS {task}: check on 24 real selection images: current {chk[0][0]:.2f} dB -> best found {chk[1][0]:.2f} dB ({chk[1][0] - chk[0][0]:+.2f})", flush=True)
