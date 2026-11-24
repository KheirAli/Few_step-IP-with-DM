"""Coloured renoising in the operator's coordinates (after FIRE), on each task's clean-protocol configuration (run from PnP-CM/).
Error tables on generated images (with the band combination in the loop), then weights refitted and the 107 test images once.
  budget : same injected power as the shipped noise, told level unchanged        exact : error + noise = the level the model is told"""
import json, subprocess, os, glob, re, sys, time, shutil, threading, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; ch = json.load(open(f"{L}/clean_protocol_choices.json"))
src = open(f"{L}/clean_protocol.py").read(); exec(src[src.index("LIN = ["):src.index("PREV = {")])
TASKS = sys.argv[1:] or ["sr_bicubic", "deblur_gauss", "inpainting", "ct30", "ct60", "hdr", "quant", "onebit", "gamma"]
def launch(t, tag, extra, g0, skiptest=False):
    c = ch[t]; deg, ex = [(d, e) for n, d, e, b in CAND[t] if n == c["variant"]][0]; cf = f"{deg} {ex} SIGS={','.join(f'{v:.4f}' for v in c['sig'])} SIGPOS=0,1,2,3 SIGTOLD={c['told']} {extra}"
    env = dict(os.environ, TAG=f"{tag}{t}", KEY=t, CFGSTR=cf, NSTEP="4", MODES="zband8", CALSH="0:64 64:128", NGPU="8", PAR="2", GOFF=str(g0))
    if skiptest: env["SKIPTEST"] = "1"
    return subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=env, stdout=open(f"{L}/logs_x/launch_{tag}{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L)
def one(t, g0):
    ps = []
    for k, m in enumerate(("budget", "exact")):
        sd = f"spec_fire_{m}_{t}"; shutil.rmtree(f"{L}/{sd}", ignore_errors=True); ps.append((m, sd, launch(t, f"fc{m[0]}", f"TILT=0 NOISE=firecal FIRE_MODE={m} SPECDIR={sd}", g0 + 2 * k, skiptest=True)))
    q = []
    for k, (m, sd, p) in enumerate(ps): p.wait(); q.append(launch(t, f"fr{m[0]}", f"TILT=0 NOISE=fire FIRE_MODE={m} SPECDIR={sd}", g0 + 2 * k))
    [p.wait() for p in q]
th = []
for i, t in enumerate(TASKS):
    while len([x for x in th if x.is_alive()]) >= int(os.environ.get("NPAR", "3")): time.sleep(10)
    x = threading.Thread(target=one, args=(t, 3 * i)); x.start(); th.append(x)
[x.join() for x in th]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def sc(pat):
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{pat}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return f"{np.mean(Ps):.2f}/{np.mean(Ls):.3f}" + ("" if len(Ps) == 107 else f" (n={len(Ps)})") if Ps else "-"
print("| Task | White | Tilt 0.5 (current) | Operator-domain renoising, same injected power | Operator-domain renoising, total = told level |\n|---|---|---|---|---|", flush=True)
for t in TASKS: print(f"| {t} | {sc(f'naW{t}test_zband8_{t}_*')} | {sc(f'cln{t}test_zband8_{t}_*')} | {sc(f'frb{t}test_zband8_{t}_*')} | {sc(f'fre{t}test_zband8_{t}_*')} |", flush=True)
