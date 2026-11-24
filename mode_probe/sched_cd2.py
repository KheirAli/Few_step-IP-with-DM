"""Schedule tuning WITH the combination in the objective, and with the number of steps as part of the search.
Objective per candidate (one sampler run on 48 model-generated images, no extra runs):
  save every step's model output and the last data-fit image, fit ONE final combination (per frequency band, 8 bands) by least squares,
  score it by two-fold cross-validation (fit on 24 images, score on the other 24, and the reverse).
For N = 4, 3, 2 steps: one pass of coordinate search over the N noise levels (x0.3, x0.55, x1.8, x3.3), then the told-level factor (1.0, 1.5).
usage (from PnP-CM/): python ../mode_probe/sched_cd2.py <task> <gpu offset>"""
import sys, os, glob, re, json, subprocess, shutil, numpy as np, torch
from PIL import Image
task = sys.argv[1]; OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 0; L = "../mode_probe"; NI = 48
src = open(f"{L}/sched_search.py").read(); exec(src[src.index("CFG = {"):src.index("deg, extra, base")])
if os.path.exists(f"{L}/sched_cfg_extra.json"): CFG.update({k_: tuple(v_) for k_, v_ in json.load(open(f"{L}/sched_cfg_extra.json")).items()})
deg, extra, base = CFG[task]; DATA = os.path.realpath(f"{L}/data_gen512"); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f_ = torch.fft.fftfreq(256) * 256; BI = torch.bucketize((f_[:, None] ** 2 + f_[None, :] ** 2).sqrt(), torch.tensor([1.5 * 2 ** k for k in range(7)])).reshape(-1).repeat(3)
def score(tag, n, N):
    H = []; G = []; P0 = []
    for rf in sorted(glob.glob(f"{L}/exp_fin_{tag}_s0/raw_*.pt")):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); R = torch.load(rf); S = {k: v[0].float() for _, k, v in R["steps"]}
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
        if task.startswith("prmask"): gt = gt * torch.sign((R["x_raw"][0].float() * gt).sum(dim=(1, 2), keepdim=True))      # channel signs are not identifiable
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
def run(cands, name, data=DATA, n=NI):
    pr = []
    for k, (sig, told, pos) in enumerate(cands):
        tag = f"cd2_{task}_{name}_{k}"; env = dict(os.environ, CUDA_VISIBLE_DEVICES=str((OFF + k) % 8), DATA=data, SHARD=f"0:{n}", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", SAVE_STEPS="1", LLE="/nonexistent_weights.json",
                                                    SIGS=",".join(f"{v:.4f}" for v in sig), SIGPOS=",".join(f"{v:g}" for v in pos), SIGTOLD=f"{told:.3f}", OMP_NUM_THREADS="4"); env.update(dict(kv.split("=", 1) for kv in extra.split()))
        pr.append((tag, len(sig), subprocess.Popen(["../pnpcm_env/bin/python", f"{L}/unified.py", f"fin_{tag}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)))
    out = []
    for tag, N, p in pr: p.wait(); out.append(score(tag, n, N)); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
    return out
res = {}
for N, pos in [c_ for c_ in ((4, [0, 1, 2, 3]), (3, [0, 1.5, 3]), (2, [0, 3])) if str(c_[0]) in os.environ.get("NSTEPS", "4,3,2").split(",")]:
    cur = ([float(np.interp(p, [0, 1, 2, 3], base)) for p in pos], 1.2, pos); b0 = run([cur], f"n{N}b")[0]; best = b0[0]; nr = 1
    for k in range(N):
        cs = []
        for m in (0.3, 0.55, 1.8, 3.3):
            s = list(cur[0]); s[k] = max(s[k], 0.05) * m; cs.append((s, cur[1], pos))
        r = run(cs, f"n{N}s{k}"); nr += 4; j = int(np.nanargmax([v[0] for v in r]))
        if r[j][0] > best: best = r[j][0]; cur = cs[j]
    cs = [(cur[0], 1.0, pos), (cur[0], 1.5, pos)]; r = run(cs, f"n{N}t"); nr += 2; j = int(np.nanargmax([v[0] for v in r]))
    if r[j][0] > best: best = r[j][0]; cur = cs[j]
    res[N] = {"sig": cur[0], "told": cur[1], "pos": pos, "start": b0, "best": best}
    print(f"CD2 {task:12s} {N} steps, {nr} runs: start {b0[0]:.2f} dB with combination ({b0[1]:.2f} without) -> tuned {best:.2f} dB | sigma " + " ".join(f"{v:.3f}" for v in cur[0]) + f" told x{cur[1]}", flush=True)
json.dump(res, open(f"{L}/sched_cd2_{task}.json", "w"))
