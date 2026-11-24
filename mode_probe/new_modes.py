"""New combination modes against the current one (8 bands + last data-fit image), on each task's clean-protocol configuration (run from PnP-CM/):
  zbandH8 : 8 bands, ALL data-fit images so far as terms          zreg8 : 8 bands, separate weights on well-measured pixels and on the rest
Weights fitted on 128 generated images; 107 test images once."""
import json, subprocess, os, glob, re, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; ch = json.load(open(f"{L}/clean_protocol_choices.json"))
src = open(f"{L}/clean_protocol.py").read(); exec(src[src.index("LIN = ["):src.index("PREV = {")])
T = {"hdr": "zbandH8 zreg8", "inpainting": "zbandH8 zreg8", "blurclip": "zbandH8 zreg8", "sr_bicubic": "zbandH8", "quant": "zbandH8", "onebit": "zbandH8", "jpeg": "zbandH8", "deblur_gauss": "zbandH8", "gamma": "zbandH8"}
run = []
for i, (t, modes) in enumerate(T.items()):
    while len([p for p in run if p.poll() is None]) >= 5: time.sleep(5)
    c = ch[t]; deg, ex = [(d, e) for n, d, e, b in CAND[t] if n == c["variant"]][0]; cfg = f"{deg} {ex} SIGS={','.join(f'{v:.4f}' for v in c['sig'])} SIGPOS=0,1,2,3 SIGTOLD={c['told']}"
    run.append(subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=f"nm{t}", KEY=t, CFGSTR=cfg, NSTEP="4", MODES=modes, CALSH="0:64 64:128", NGPU="8", PAR="4", GOFF=str(2 * i)), stdout=open(f"{L}/logs_x/launch_nm{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L))
[p.wait() for p in run]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def score(pat):
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{pat}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return f"{np.mean(Ps):.2f}/{np.mean(Ls):.3f}" + ("" if len(Ps) == 107 else f" (n={len(Ps)})") if Ps else "-"
print("| Task | 8 bands + last data-fit image (current) | 8 bands + all data-fit images | 8 bands x measurement region |\n|---|---|---|---|", flush=True)
for t in T: print(f"| {t} | {score(f'cln{t}test_zband8_{t}_*')} | {score(f'nm{t}test_zbandH8_{t}_*')} | {score(f'nm{t}test_zreg8_{t}_*')} |", flush=True)
