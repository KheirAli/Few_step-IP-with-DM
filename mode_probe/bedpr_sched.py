"""4-step schedule tuning for BEDROOM unmasked phase retrieval: sched_cd2.py's search and objective, unchanged (one pass of coordinate search over
the 4 noise levels x0.3/x0.55/x1.8/x3.3, then the told factor 1.0/1.5; objective = two-fold cross-validated 8-band combination of the step outputs on
48 MODEL-GENERATED bedroom images, data_bedgen 0:48).  Differences from sched_cd2.py are operational only: DSET=bedroom, tags start with bedpr,
at most MAXP sampler processes at a time, each on the GPU with the most free memory (>= 8 GB), rerun if a run dies (out of memory).
usage (from PnP-CM/): python ../mode_probe/bedpr_sched.py <free|direct>      -> ../mode_probe/bedpr_sched_<task>.json"""
import sys, os, glob, re, json, subprocess, shutil, time, numpy as np, torch
from PIL import Image
task = sys.argv[1]; L = "../mode_probe"; NI = 48; MAXP = int(os.environ.get("MAXP", "2")); gs = lambda a, n: [float(v) for v in np.geomspace(a, 0.03, n)]
B = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1"
CFG = {"free": ("deblur_nl", f"{B} PRD_TV=80 PRD_INIT=regfit4 PRD_NSTART=16 TILT=0.5", gs(1.0, 4)),                       # = face key pru_t05
       "direct": ("deblur_nl", f"{B} PRD_NET=v2k8 PRD_INIT=netdirect PRD_TV=0 PRD_NETR=32 TILT=0.5", gs(0.7, 4))}         # = face key prd_t05
deg, extra, base = CFG[task]; DATA = os.path.realpath(f"{L}/data_bedgen")
f_ = torch.fft.fftfreq(256) * 256; BI = torch.bucketize((f_[:, None] ** 2 + f_[None, :] ** 2).sqrt(), torch.tensor([1.5 * 2 ** k for k in range(7)])).reshape(-1).repeat(3)
def score(tag, n, N):
    H = []; G = []; P0 = []
    for rf in sorted(glob.glob(f"{L}/exp_fin_{tag}_s0/raw_*.pt")):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); R = torch.load(rf); S = {k: v[0].float() for _, k, v in R["steps"]}
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
        H.append(torch.stack([S[100 + j] for j in range(N)] + [S[200 + N - 1]])); G.append(gt); P0.append(float(10 * torch.log10(1 / (((R["x_raw"][0].float().clamp(-1, 1) - gt) / 2) ** 2).mean())))
    if len(H) != n: return float("nan"), float("nan")
    Hf = torch.fft.fft2(torch.stack(H), norm="ortho").flatten(2); Gf = torch.fft.fft2(torch.stack(G), norm="ortho").flatten(1); T = Hf.shape[1]; idx = torch.arange(n); out = []
    for fold in (0, 1):
        tr = idx % 2 == fold; te = ~tr; W = torch.zeros(T, 8, dtype=torch.cfloat)
        for b in range(8):
            m = BI == b; A = Hf[tr][:, :, m].permute(0, 2, 1).reshape(-1, T); y = Gf[tr][:, m].reshape(-1)
            Gm = (A.conj().T @ A).real.double(); W[:, b] = (torch.linalg.pinv(Gm + 1e-6 * Gm.diagonal().mean() * torch.eye(T, dtype=torch.float64), hermitian=True) @ (A.conj().T @ y).real.double()).float()
        rec = torch.fft.ifft2((W[:, BI][None] * Hf[te]).sum(1).view(-1, 3, 256, 256), norm="ortho").real.clamp(-1, 1); gte = torch.stack(G)[te]
        out += [float(v) for v in 10 * torch.log10(1 / (((rec - gte) / 2) ** 2).mean(dim=(1, 2, 3)))]
    return float(np.mean(out)), float(np.mean(P0))
def launch(tag, sig, told, pos):
    g = subprocess.run(["bash", f"{L}/bedpr_pick.sh", "8000"], capture_output=True, text=True).stdout.strip(); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, DATA=DATA, SHARD=f"0:{NI}", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", SAVE_STEPS="1", LLE="/nonexistent_weights.json",
               DSET="bedroom", MEANFACE="mean_bed_gen.pt", SIGS=",".join(f"{v:.4f}" for v in sig), SIGPOS=",".join(f"{v:g}" for v in pos), SIGTOLD=f"{told:.3f}", OMP_NUM_THREADS="4"); env.update(dict(kv.split("=", 1) for kv in extra.split()))
    return subprocess.Popen(["../pnpcm_env/bin/python", f"{L}/unified.py", f"fin_{tag}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)
def run(cands, name):
    todo = [(f"bedprcd2_{task}_{name}_{k}", c, 0) for k, c in enumerate(cands)]; act = []; done = {}
    while todo or act:
        for it in list(act):
            tag, c, tr, p = it
            if p.poll() is None: continue
            act.remove(it); ok = len(glob.glob(f"{L}/exp_fin_{tag}_s0/raw_*.pt")) == NI
            if ok or tr >= 6: done[tag] = score(tag, NI, len(c[0])); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
            else: print(f"   rerun {tag} (try {tr + 2})", flush=True); todo.append((tag, c, tr + 1))
        if todo and len(act) < MAXP:
            tag, c, tr = todo.pop(0); act.append((tag, c, tr, launch(tag, *c))); time.sleep(40); continue
        time.sleep(5)
    return [done[f"bedprcd2_{task}_{name}_{k}"] for k in range(len(cands))]
res = {}; N = 4; pos = [0, 1, 2, 3]
cur = (list(base), 1.2, pos); b0 = run([cur], f"n{N}b")[0]; best = b0[0]; nr = 1; print(f"BEDSCHED {task} start {b0[0]:.2f} ({b0[1]:.2f} without combination)", flush=True)
for k in range(N):
    cs = []
    for m in (0.3, 0.55, 1.8, 3.3):
        s = list(cur[0]); s[k] = max(s[k], 0.05) * m; cs.append((s, cur[1], pos))
    r = run(cs, f"n{N}s{k}"); nr += 4; j = int(np.nanargmax([v[0] for v in r])); print(f"BEDSCHED {task} level {k}: " + " | ".join(f"x{m}: {v[0]:.2f}" for m, v in zip((0.3, 0.55, 1.8, 3.3), r)) + f" (current {best:.2f})", flush=True)
    if r[j][0] > best: best = r[j][0]; cur = cs[j]
cs = [(cur[0], 1.0, pos), (cur[0], 1.5, pos)]; r = run(cs, f"n{N}t"); nr += 2; j = int(np.nanargmax([v[0] for v in r])); print(f"BEDSCHED {task} told: x1.0 {r[0][0]:.2f} | x1.5 {r[1][0]:.2f}", flush=True)
if r[j][0] > best: best = r[j][0]; cur = cs[j]
res[N] = {"sig": cur[0], "told": cur[1], "pos": pos, "start": b0, "best": best}
print(f"BEDSCHED {task:8s} {N} steps, {nr} runs: start {b0[0]:.2f} dB with combination ({b0[1]:.2f} without) -> tuned {best:.2f} dB | sigma " + " ".join(f"{v:.3f}" for v in cur[0]) + f" told x{cur[1]}", flush=True)
json.dump(res, open(f"{L}/bedpr_sched_{task}.json", "w"))
