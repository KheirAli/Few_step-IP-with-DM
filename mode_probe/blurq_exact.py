"""Blur then 3-bit quantisation: per-coordinate posterior mean and a version without alternation, compared on GENERATED images only
(weights on generated 0:128, score on generated 128:176), then the chosen one on the 107 test images once (run from PnP-CM/)."""
import json, subprocess, os, glob, re, sys, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; c = json.load(open(f"{L}/clean_protocol_choices.json"))["blurq"]; SCH = f"SIGS={','.join(f'{v:.4f}' for v in c['sig'])} SIGPOS=0,1,2,3 SIGTOLD={c['told']}"; B = "NEWOP=blurq"
V = {"nearest level, 8 alternations (current)": B, "posterior mean, 8 alternations": f"{B} SPLIT_EV=1", "posterior mean (T x0.3), 8 alternations": f"{B} SPLIT_EV=0.3",
     "posterior mean, no alternation, coupling 2": f"{B} SPLIT_EV=1 SPLIT_N=1 SPLIT_MU=2", "posterior mean, no alternation, coupling 8": f"{B} SPLIT_EV=1 SPLIT_N=1 SPLIT_MU=8",
     "nearest level, no alternation, coupling 2": f"{B} SPLIT_N=1 SPLIT_MU=2", "nearest level, no alternation, coupling 8": f"{B} SPLIT_N=1 SPLIT_MU=8"}
GP = os.environ.get("GPULIST", "0 4 6 7")
def launch(k, ex, test, sh, g0): return subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=f"bq{k}", KEY="blurq", CFGSTR=f"deblur_nl {ex} {SCH}", NSTEP="4", MODES="zband8", CALSH="0:64 64:128", TESTDATA=test, TESTSH=sh, NGPU="8", PAR="2", GOFF=str(g0), GPULIST=GP), stdout=open(f"{L}/logs_x/launch_bq{k}.log", "w"), stderr=subprocess.STDOUT, cwd=L)
lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def sc(pat):
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{pat}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return (float(np.mean(Ps)), float(np.mean(Ls)), len(Ps)) if Ps else (float("nan"), float("nan"), 0)
ps = [launch(k, ex, f"{L}/data_gen512", "128:152 152:176", k) for k, ex in enumerate(V.values())]; [p.wait() for p in ps]; R = {}
for k, n in enumerate(V): R[n] = sc(f"bq{k}test_zband8_blurq_deblur_nl"); print(f"BQ generated images: {n}: {R[n][0]:.2f}/{R[n][1]:.3f} (n={R[n][2]})", flush=True)
ok = {n: r for n, r in R.items() if r[2] == 48}; b = max(ok, key=lambda n: ok[n][0]); na = {n: r for n, r in ok.items() if "no alternation" in n}; b1 = max(na, key=lambda n: na[n][0]) if na else None
print(f"BQ chosen on generated images: {b} | best without alternation: {b1}", flush=True); subprocess.run(f"rm -rf {L}/exp_fin_bq*test_*", shell=True)
T = {"T": b}
if b1 and b1 != b: T["U"] = b1
ps = [launch(k, V[n], f"{L}/data_paper", "0:54 54:107", 2 * i) for i, (k, n) in enumerate(T.items())]; [p.wait() for p in ps]
for k, n in T.items(): r = sc(f"bq{k}test_zband8_blurq_deblur_nl"); print(f"BQ TEST {n}: {r[0]:.2f}/{r[1]:.3f} (n={r[2]}) | current 26.62/0.351 | DAPS 24.76/0.388 | DPS 24.63/0.276", flush=True)
