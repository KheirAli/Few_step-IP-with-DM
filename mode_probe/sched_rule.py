"""Self-consistent schedule rule: sigma_k = c * (rms error of the model input at step k), error measured on model-generated images.
Start from the current schedule, re-measure and reset 3 times (3 short runs per value of c), then score on the 24 real selection images.
usage (from PnP-CM/): python ../mode_probe/sched_rule.py"""
import json, subprocess, os, glob, re, numpy as np, torch
from PIL import Image
L = "../mode_probe"
CFG = {"sr_bicubic": ("sr_bicubic", "LINEXACT=1", [0.465, 0.245, 0.01, 0.01]), "deblur_gauss": ("deblur_gauss", "LINEXACT=1", [0.327, 0.132, 0.01, 0.01]), "inpainting": ("inpainting", "LINEXACT=1 FW=1 FWP=0.5", [0.327, 0.132, 0.01, 0.01]),
       "hdr": ("deblur_nl", "NEWOP=hdr HDRX=1", [0.694, 0.635, 0.56, 0.459]), "quant": ("deblur_nl", "NEWOP=quant FW=1 FWP=0.5 FWMIN=0.4", [0.694, 0.635, 0.56, 0.459]),
       "onebit": ("deblur_nl", "NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2 FW=1 FWP=0.5 FWMIN=0.4", [0.694, 0.635, 0.56, 0.459])}
CS = [float(v) for v in os.environ.get("CS", "1.0,1.5,2.0,3.0").split(",")]; TOLD = float(os.environ.get("TOLD", "1.2"))
def launch(tag, t, sig, data, n, g, diag):
    deg, extra, _ = CFG[t]; env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % 8), DATA=os.path.realpath(data), SHARD=f"0:{n}", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5",
                                       DIAG3="1" if diag else "0", SIGS=",".join(f"{v:.4f}" for v in sig), SIGPOS="0,1,2,3", SIGTOLD=f"{TOLD:.3f}")
    env.update(dict(kv.split("=", 1) for kv in extra.split()))
    return subprocess.Popen(["../pnpcm_env/bin/python", f"{L}/unified.py", f"fin_{tag}", f"--deg={deg}"], env=env, stdout=open(f"{L}/logs_x/{tag}.out", "w"), stderr=subprocess.STDOUT)
def psnr_of(tag):
    P = []
    for rf in glob.glob(f"{L}/exp_fin_{tag}_s0/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
        gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1); P.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean())))
    return float(np.mean(P))
chains = {(t, c): list(CFG[t][2]) for t in CFG for c in CS}
for it in range(3):
    pr = {k: launch(f"rule_{k[0]}_{k[1]}_{it}", k[0], s, f"{L}/data_gen", 24, g, True) for g, (k, s) in enumerate(chains.items())}
    for k, p in pr.items():
        p.wait(); D = np.array([s["diag3"] for s in json.load(open(f"{L}/exp_fin_rule_{k[0]}_{k[1]}_{it}_s0/stats.json")) if "diag3" in s]).mean(0)
        chains[k] = [float(np.clip(k[1] * e, 0.01, 20.0)) for e in np.sqrt(D[:, 0].sum(1))]
pr = {k: launch(f"rule_{k[0]}_{k[1]}_fin", k[0], s, f"{L}/data100", 24, g, False) for g, (k, s) in enumerate(chains.items())}
res = {}
for k, p in pr.items(): p.wait(); res[k] = psnr_of(f"rule_{k[0]}_{k[1]}_fin")
for t in CFG:
    print(f"RULE2 {t:12s} told x{TOLD} | " + " | ".join(f"c={c}: {res[(t, c)]:.2f} dB (sigma " + " ".join(f"{v:.2f}" for v in chains[(t, c)]) + ")" for c in CS), flush=True)
