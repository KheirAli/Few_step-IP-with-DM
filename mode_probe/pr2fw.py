"""Masked phase retrieval, attempt 2 with a frequency-weighted arm, so that it is like for like with attempt 1 (run from PnP-CM/).
Candidates: coupling weight 0.1 / 0.3, with and without the frequency-weighted fit; schedule tuned; generated mean face; choices on generated images."""
import json, subprocess, os, glob, re, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; PRM = "INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=prior LINEXACT=1 PR2=1"; LIN = [0.327, 0.132, 0.01, 0.01]
K = {f"prmask__p2_{r}_{f}": ["phase_retrieval", f"{PRM} PR2_RS={r}" + (" FW=1 FWP=0.5" if f == "fw" else ""), LIN] for r in ("0.1", "0.3") for f in ("plain", "fw")}
ex = json.load(open(f"{L}/sched_cfg_extra.json")); ex.update(K); json.dump(ex, open(f"{L}/sched_cfg_extra.json", "w"))
sp = [subprocess.Popen([P, f"{L}/sched_cd2.py", k, str(2 * g)], env=dict(os.environ, NSTEPS="4"), stdout=open(f"{L}/logs_x/pr2fw_cd2_{k}.log", "w"), stderr=subprocess.STDOUT) for g, k in enumerate(K)]; [p.wait() for p in sp]
R = {k: json.load(open(f"{L}/sched_cd2_{k}.json"))["4"] for k in K if os.path.exists(f"{L}/sched_cd2_{k}.json")}; print("P2 tuning on generated images: " + " | ".join(f"{k.split('__')[1]}: {v['start'][0]:.2f} -> {v['best']:.2f}" for k, v in R.items()), flush=True)
b = max(R, key=lambda k: R[k]["best"]); r = R[b]; cfg = f"phase_retrieval {K[b][1]} SIGS={','.join(f'{v:.4f}' for v in r['sig'])} SIGPOS=0,1,2,3 SIGTOLD={r['told']}"; print("P2 chosen:", b, flush=True)
subprocess.run(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG="p2fw", KEY="prmask", CFGSTR=cfg, NSTEP="4", MODES="zband8", CALSH="0:64 64:128", NGPU="8", PAR="4", GOFF="0"), stdout=open(f"{L}/logs_x/launch_p2fw.log", "w"), stderr=subprocess.STDOUT, cwd=L)
lp = lpips.LPIPS(net="vgg", verbose=False).cuda(); Ps = []; Ls = []
for rf in glob.glob(f"{L}/exp_fin_p2fwtest_zband8_prmask_phase_retrieval_s*/raw_*.pt"):
    d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
    with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
print(f"P2 TEST masked phase retrieval, attempt 2 ({b.split('__')[1]}), as produced: {np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)}) | attempt 1 (reduction to inpainting, generated mean face) 28.38/0.270", flush=True)
