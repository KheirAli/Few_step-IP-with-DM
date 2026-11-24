"""Rerun the two results that used the mean face of REAL images (mean_face01.pt = mean of the 100 data100 images, two of which are test images)
with the mean of 512 MODEL-GENERATED images (mean_face_gen.pt, now the default): masked phase retrieval (sign rule) and coded diffraction (start)."""
import json, subprocess, os, glob, re, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; ch = json.load(open(f"{L}/clean_protocol_choices.json"))["prmask"]
sg = lambda r: f"SIGS={','.join(f'{v:.4f}' for v in r['sig'])} SIGPOS=0,1,2,3 SIGTOLD={r['told']}"
J = [("mfM", "prmask", f"phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=prior LINEXACT=1 FW=1 FWP=0.5 {sg(ch)}"),
     ("mfC", "cdp2", "deblur_nl NEWOP=prcdp PR_MASKS=2 RHOX=0.3 PR_ISO_WARM=50 SIGS=0.1500,0.1375,0.1800,0.0900 SIGPOS=0,1,2,3 SIGTOLD=1.5")]
print("configs:", [j[2][-70:] for j in J], flush=True)
run = [subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=t, KEY=k, CFGSTR=c, NSTEP="4", MODES="zband8", CALSH="0:64 64:128", NGPU="8", PAR="4", GOFF=str(4 * i)), stdout=open(f"{L}/logs_x/launch_{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L) for i, (t, k, c) in enumerate(J)]
[p.wait() for p in run]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
for (t, k, c), prev in zip(J, ("28.20/0.273", "34.45/0.126")):
    Ps = []; Ls = []; ok = []
    for d in glob.glob(f"{L}/exp_fin_{t}test_zband8_{k}_*_s*"):
        if os.path.exists(d + "/stats.json"): ok += [s["sign_ok"] for s in json.load(open(d + "/stats.json")) if "sign_ok" in s]
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1; x = torch.load(rf)["x_raw"].float().clamp(-1, 1)
            with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x.cuda(), gt.cuda())))
    print(f"MF {k}: generated mean face {np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)})" + (f", channel signs right {100 * np.mean(ok):.1f}%" if ok else "") + f" | with the real-image mean face {prev}", flush=True)
