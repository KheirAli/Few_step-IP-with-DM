"""Unmasked phase retrieval, CLEAN PROTOCOL (run from PnP-CM/): the choices previously made on test images are re-made on 24 MODEL-GENERATED images
(one measurement-noise draw): classical-loop length, offset on/off, starting noise level of the schedule, and (learned variant) the predictor radius.
Then the chosen settings are run once on the 107 test images with 3 measurement-noise draws."""
import subprocess, os, glob, re, json, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; gs = lambda a, n: ",".join("%.4f" % v for v in np.geomspace(a, 0.03, n))
BASE = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1"
def launch(tag, cfg, data, shard, ms, g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % 8), DATA=os.path.realpath(data), SHARD=shard, SAMPLE_SEED="1", MEAS_SEED=str(ms), RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0", OMP_NUM_THREADS="4")
    env.update(dict(kv.split("=", 1) for kv in (BASE + " " + cfg).split()))
    return subprocess.Popen([P, f"{L}/unified.py", f"fin_{tag}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)
def pool(jobs, mx=10):
    run = []
    for g, j in enumerate(jobs):
        while len([p for p in run if p.poll() is None]) >= mx: time.sleep(3)
        run.append(launch(*j, g))
    [p.wait() for p in run]
def psnr(tag):
    f = glob.glob(f"{L}/exp_fin_{tag}_s0/stats.json"); p = [s["psnr"] for s in json.load(open(f[0]))] if f else []
    return float(np.mean(p)) if len(p) >= 20 else float("nan")
GEN = f"{L}/data_gen512"; res = {}
# stage 1 (training-free): loop length x offset, shipped schedules
C1 = {f"tv{tv}_off{o}": f"PRD_INIT=regfit4 PRD_NSTART=16 PRD_OFFSET={o} PRD_TV={tv}" for tv in (0, 20, 40, 80) for o in (0, 1)}
pool([(f"prc_{n}_{st}", f"{c} SIGS={gs(0.7, 4) if st == 4 else gs(1.5, 8)}", GEN, "0:24", 0) for n, c in C1.items() for st in (4, 8)])
s1 = {n: (psnr(f"prc_{n}_4"), psnr(f"prc_{n}_8")) for n in C1}; b1 = max(s1, key=lambda n: np.nansum(s1[n])); print("PRC stage 1 (4 / 8 calls): " + " | ".join(f"{n}: {v[0]:.2f}/{v[1]:.2f}" for n, v in s1.items()) + f" || chosen {b1}", flush=True)
# stage 2: starting noise level of the schedule
S = {4: (0.35, 0.5, 0.7, 1.0, 1.4), 8: (0.75, 1.0, 1.5, 2.0, 3.0)}
pool([(f"prc_s{st}_{a}", f"{C1[b1]} SIGS={gs(a, st)}", GEN, "0:24", 0) for st in (4, 8) for a in S[st]])
bs = {}
for st in (4, 8):
    r = {a: psnr(f"prc_s{st}_{a}") for a in S[st]}; bs[st] = max(r, key=lambda a: r[a] if r[a] == r[a] else -1); print(f"PRC stage 2, {st} calls, first noise level: " + " | ".join(f"{a}: {v:.2f}" for a, v in r.items()) + f" || chosen {bs[st]}", flush=True)
# stage 3 (learned predictor): radius
NET = "PRD_INIT=netfit PRD_NET=v2k8 PRD_NETREG=0"; tvo = C1[b1].split("PRD_NSTART=16 ")[1]
pool([(f"prc_net{r}_{st}", f"{NET} PRD_NETR={r} {tvo} SIGS={gs(bs[st], st)}", GEN, "0:24", 0) for r in (16, 32, 48) for st in (4, 8)])
rn = {r: (psnr(f"prc_net{r}_4"), psnr(f"prc_net{r}_8")) for r in (16, 32, 48)}; br = max(rn, key=lambda r: np.nansum(rn[r])); print("PRC stage 3, predictor radius (4 / 8 calls): " + " | ".join(f"{r}: {v[0]:.2f}/{v[1]:.2f}" for r, v in rn.items()) + f" || chosen {br}", flush=True)
json.dump({"stage1": b1, "first_sigma": bs, "radius": br}, open(f"{L}/pr_clean_choices.json", "w"))
# test: 107 images x 3 measurement-noise draws
TEST = f"{L}/data_paper"
pool([(f"prct_{v}_{st}_m{ms}", (f"{C1[b1]}" if v == "free" else f"{NET} PRD_NETR={br} {tvo}") + f" SIGS={gs(bs[st], st)}", TEST, "0:107", ms) for v in ("free", "net") for st in (4, 8) for ms in (0, 1, 2)], mx=12)
lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
for v, name in (("free", "training-free"), ("net", "learned predictor")):
    for st in (4, 8):
        Ps = []; Ls = []
        for d in glob.glob(f"{L}/exp_fin_prct_{v}_{st}_m*_s0"):
            stt = {x["img"]: x for x in json.load(open(d + "/stats.json"))} if os.path.exists(d + "/stats.json") else {}
            for rf in glob.glob(d + "/raw_*.pt"):
                i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1; x = torch.load(rf)["x_raw"].float()
                with torch.no_grad(): Ps.append(stt[i]["psnr"] if i in stt else float(10 * torch.log10(1 / (((x.clamp(-1, 1) - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x.cuda(), gt.cuda())))
        Ps = np.array(Ps); print(f"PRC TEST {name}, {st} calls: {Ps.mean():.2f}/{np.mean(Ls):.3f} (n={len(Ps)}), below 14 dB {100 * (Ps < 14).mean():.1f}%" if len(Ps) else f"PRC TEST {name} {st} missing", flush=True)
