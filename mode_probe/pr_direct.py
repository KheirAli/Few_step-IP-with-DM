"""NON-ITERATIVE unmasked phase retrieval (run from PnP-CM/): one pass of the phase predictor -> coarse image by one inverse transform (measured
magnitudes + predicted phases) -> 4 or 8 model calls, each preceded by the per-coefficient magnitude fit.  No eigenface fit, no multi-start,
no classical loop.  The first noise level is chosen on 24 generated images; then 107 test images x 3 measurement-noise draws."""
import subprocess, os, glob, re, json, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; gs = lambda a, n: ",".join("%.4f" % v for v in np.geomspace(a, 0.03, n))
B = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_NET=v2k8 PRD_INIT=netdirect PRD_TV=0 PRD_NETR=32"
def launch(tag, cfg, data, shard, ms, g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % 8), DATA=os.path.realpath(data), SHARD=shard, SAMPLE_SEED="1", MEAS_SEED=str(ms), RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0", OMP_NUM_THREADS="4")
    env.update(dict(kv.split("=", 1) for kv in (B + " " + cfg).split())); return subprocess.Popen([P, f"{L}/unified.py", f"fin_{tag}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)
def psnr(tag):
    f = glob.glob(f"{L}/exp_fin_{tag}_s0/stats.json"); p = [s["psnr"] for s in json.load(open(f[0]))] if f else []; return float(np.mean(p)) if len(p) >= 20 else float("nan")
S = {4: (0.5, 0.7, 1.0, 1.4, 2.0, 3.0), 8: (1.0, 1.5, 2.0, 3.0, 4.5)}
pr = [launch(f"prd_s{st}_{a}", f"SIGS={gs(a, st)}", f"{L}/data_gen512", "0:24", 0, g) for g, (st, a) in enumerate([(st, a) for st in (4, 8) for a in S[st]])]; [p.wait() for p in pr]; bs = {}
for st in (4, 8):
    r = {a: psnr(f"prd_s{st}_{a}") for a in S[st]}; bs[st] = max(r, key=lambda a: r[a] if r[a] == r[a] else -1); print(f"PRD generated images, {st} calls, first noise level: " + " | ".join(f"{a}: {v:.2f}" for a, v in r.items()) + f" || chosen {bs[st]}", flush=True)
pr = [launch(f"prdt_{st}_m{ms}", f"SIGS={gs(bs[st], st)}", f"{L}/data_paper", "0:107", ms, g) for g, (st, ms) in enumerate([(st, ms) for st in (4, 8) for ms in (0, 1, 2)])]; [p.wait() for p in pr]
lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
for st in (4, 8):
    Ps = []; Ls = []; Tm = []
    for d in glob.glob(f"{L}/exp_fin_prdt_{st}_m*_s0"):
        stt = {x["img"]: x for x in json.load(open(d + "/stats.json"))}; Tm += [x.get("time_total", np.nan) for x in list(stt.values())[2:]]
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1; x = torch.load(rf)["x_raw"].float()
            with torch.no_grad(): Ps.append(stt[i]["psnr"]); Ls.append(float(lp(x.cuda(), gt.cuda())))
    Ps = np.array(Ps); print(f"PRD TEST non-iterative, {st} calls: {Ps.mean():.2f}/{np.mean(Ls):.3f} (n={len(Ps)}), below 14 dB {100 * (Ps < 14).mean():.1f}%, 5th percentile {np.percentile(Ps, 5):.1f} dB, time {np.nanmedian(Tm):.2f}s", flush=True)
