"""Noise spectrum of the injected noise: four arms on each task's clean-protocol configuration (same data step, schedule, told level and band combination).
  white   : white noise                                   tilt   : power ~ (frequency rank)^0.5 (current)
  matched : spectrum computed from the measured error of the iterate, so that error + noise is white (no tuning)
  fitted  : one power gain per octave band (8), fitted on generated images (coordinate search, final band combination in the objective)
Everything is fitted on generated images; 107 test images once.  usage (from PnP-CM/): python ../mode_probe/noise_arms.py [tasks...]"""
import json, subprocess, os, glob, re, sys, time, shutil, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; ch = json.load(open(f"{L}/clean_protocol_choices.json")); GEN = os.path.realpath(f"{L}/data_gen512")
src = open(f"{L}/clean_protocol.py").read(); exec(src[src.index("LIN = ["):src.index("PREV = {")])
TASKS = sys.argv[1:] or ["sr_bicubic", "deblur_gauss", "inpainting", "jpeg", "deblur_nl", "hdr", "quant", "onebit", "gamma", "blurclip", "blurq", "ct30", "ct60"]
def cfg(t):
    c = ch[t]; deg, ex = [(d, e) for n, d, e, b in CAND[t] if n == c["variant"]][0]; return deg, f"{ex} SIGS={','.join(f'{v:.4f}' for v in c['sig'])} SIGPOS=0,1,2,3 SIGTOLD={c['told']}"
f_ = torch.fft.fftfreq(256) * 256; BI2 = torch.bucketize((f_[:, None] ** 2 + f_[None, :] ** 2).sqrt(), torch.tensor([1.5 * 2 ** k for k in range(7)])); BI = BI2.reshape(-1).repeat(3)
rk = torch.argsort(torch.argsort((f_[:, None] ** 2 + f_[None, :] ** 2).sqrt().flatten())).float().view(256, 256); TILT0 = [float((((rk[BI2 == b] + 0.5) / 65536) ** 0.5).mean()) for b in range(8)]   # the tilt, per octave band
def score(tag, n=48, N=4):
    H = []; G = []
    for rf in sorted(glob.glob(f"{L}/exp_fin_{tag}_s0/raw_*.pt")):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); S = {k: v[0].float() for _, k, v in torch.load(rf)["steps"]}
        G.append(torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1); H.append(torch.stack([S[100 + j] for j in range(N)] + [S[200 + N - 1]]))
    if len(H) != n: return float("nan")
    Hf = torch.fft.fft2(torch.stack(H), norm="ortho").flatten(2); Gf = torch.fft.fft2(torch.stack(G), norm="ortho").flatten(1); T = Hf.shape[1]; idx = torch.arange(n); out = []
    for fold in (0, 1):
        tr = idx % 2 == fold; te = ~tr; W = torch.zeros(T, 8, dtype=torch.cfloat)
        for b in range(8):
            m = BI == b; A = Hf[tr][:, :, m].permute(0, 2, 1).reshape(-1, T); y = Gf[tr][:, m].reshape(-1); Gm = (A.conj().T @ A).real.double()
            W[:, b] = (torch.linalg.pinv(Gm + 1e-6 * Gm.diagonal().mean() * torch.eye(T, dtype=torch.float64), hermitian=True) @ (A.conj().T @ y).real.double()).float()
        rec = torch.fft.ifft2((W[:, BI][None] * Hf[te]).sum(1).view(-1, 3, 256, 256), norm="ortho").real.clamp(-1, 1); out += [float(v) for v in 10 * torch.log10(1 / (((rec - torch.stack(G)[te]) / 2) ** 2).mean(dim=(1, 2, 3)))]
    return float(np.mean(out))
def gen_runs(t, cands, name, g0):
    deg, ex = cfg(t); pr = []
    for k, nbg in enumerate(cands):
        tag = f"nf_{t}_{name}_{k}"; env = dict(os.environ, CUDA_VISIBLE_DEVICES=str((g0 + k) % 8), DATA=GEN, SHARD="0:48", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0", NBG=",".join(f"{v:.4f}" for v in nbg), SAVE_STEPS="1", LLE="/nonexistent_weights.json", OMP_NUM_THREADS="4")
        env.update(dict(kv.split("=", 1) for kv in ex.split())); pr.append((tag, subprocess.Popen([P, f"{L}/unified.py", f"fin_{tag}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)))
    out = []
    for tag, p in pr: p.wait(); out.append(score(tag)); shutil.rmtree(f"{L}/exp_fin_{tag}_s0", ignore_errors=True)
    return out
def fit_gains(t, g0):
    """One round of coordinate search from the tilt's band profile: each band's power x0.4 and x2.5 (16 runs), then all improving moves together."""
    cs = [list(TILT0)] + [[v * (m if j == b else 1) for j, v in enumerate(TILT0)] for b in range(8) for m in (0.4, 2.5)]; r = gen_runs(t, cs, "a", g0); base = r[0]; mv = list(TILT0)
    for b in range(8):
        j = 1 + 2 * b + int(np.nanargmax(r[1 + 2 * b:3 + 2 * b]))
        if r[j] > base + 0.01: mv[b] = cs[j][b]
    rc = gen_runs(t, [mv], "c", g0)[0]; allc = cs + [mv]; alls = r + [rc]; j = int(np.nanargmax(alls)); return allc[j], base, alls[j]
def launch(t, arm, extra, g0, skiptest=False, tagx=""):
    deg, ex = cfg(t); env = dict(os.environ, TAG=f"na{arm}{tagx}{t}", KEY=t, CFGSTR=f"{deg} {ex} {extra}", NSTEP="4", MODES="zband8", CALSH="0:64 64:128", NGPU="8", PAR="2", GOFF=str(g0))
    if skiptest: env["SKIPTEST"] = "1"
    return subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=env, stdout=open(f"{L}/logs_x/launch_na{arm}{tagx}{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L)
def one_task(t, g0):
    # white and matched-calibration first (the latter measures the error spectrum on generated images, with the combination in the loop), then matched; fitted gains in parallel
    sd = f"spec_m_{t}"; shutil.rmtree(f"{L}/{sd}", ignore_errors=True)
    a = launch(t, "W", "TILT=0", g0); b = launch(t, "M", f"TILT=0 NOISE=calib SPECDIR={sd}", g0 + 2, skiptest=True, tagx="c")
    nbg, gb, gf = fit_gains(t, g0 + 4); print(f"NA {t}: fitted band gains (power, band 0..7) " + " ".join(f"{v / np.mean(nbg):.2f}" for v in nbg) + f" | generated images: tilt profile {gb:.2f} -> fitted {gf:.2f}", flush=True)
    b.wait(); c = launch(t, "M", f"TILT=0 NOISE=shape SPECDIR={sd}", g0 + 2); d = launch(t, "F", "TILT=0 NBG=" + ",".join(f"{v:.4f}" for v in nbg), g0 + 4); [p.wait() for p in (a, c, d)]
import threading
th = []
for i, t in enumerate(TASKS):
    while len([x for x in th if x.is_alive()]) >= int(os.environ.get("NPAR", "3")): time.sleep(10)
    x = threading.Thread(target=one_task, args=(t, 2 * i)); x.start(); th.append(x)
[x.join() for x in th]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def sc(pat):
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{pat}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return f"{np.mean(Ps):.2f}/{np.mean(Ls):.3f}" + ("" if len(Ps) == 107 else f" (n={len(Ps)})") if Ps else "-"
print("| Task | White | Tilt 0.5 (current) | Matched to measured error | Fitted band gains |\n|---|---|---|---|---|", flush=True)
for t in TASKS: print(f"| {t} | {sc(f'naW{t}test_zband8_{t}_*')} | {sc(f'cln{t}test_zband8_{t}_*')} | {sc(f'naM{t}test_zband8_{t}_*')} | {sc(f'naF{t}test_zband8_{t}_*')} |", flush=True)
