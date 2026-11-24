"""JPEG: variants of the exact data step, compared on GENERATED images only (weights on generated 0:128, score on generated 128:176), then the
chosen one on the 107 test images once (run from PnP-CM/).  Schedule = the clean-protocol JPEG schedule in every variant."""
import json, subprocess, os, glob, re, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; c = json.load(open(f"{L}/clean_protocol_choices.json"))["jpeg"]; SCH = f"SIGS={','.join(f'{v:.4f}' for v in c['sig'])} SIGPOS=0,1,2,3 SIGTOLD={c['told']}"
JF = os.path.realpath(f"{L}/jpeg_ev_table.json"); FW = "FW=1 FWP=0.5 FWSTEPS=1,2,3"; B = "JPROJ=1 JCOLD=1"
V = {"soft_fw (current)": f"{B} JSOFT=1 {FW}", "soft_fw + final projection": f"{B} JSOFT=1 {FW} JFINAL=1", "hard_fw + final projection": f"{B} {FW} JFINAL=1", "hard + final projection": f"{B} JFINAL=1",
     "posterior mean": f"{B} JEV=use JEVFILE={JF}", "posterior mean + final projection": f"{B} JEV=use JEVFILE={JF} JFINAL=1", "posterior mean x0.5 + final projection": f"{B} JEV=use JEVS=0.5 JEVFILE={JF} JFINAL=1"}
if os.path.exists(JF): os.remove(JF)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", DATA=os.path.realpath(f"{L}/data_gen512"), SHARD="0:48", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", JEV="cal", JEVFILE=JF, OMP_NUM_THREADS="4")
env.update(dict(kv.split("=", 1) for kv in f"{B} {SCH}".split())); subprocess.run([P, f"{L}/unified.py", "fin_jxcal", "--deg=jpeg"], env=env, stdout=open(f"{L}/logs_x/jxcal.out", "w"), stderr=subprocess.STDOUT); subprocess.run(f"rm -rf {L}/exp_fin_jxcal_s0", shell=True)
T = json.load(open(JF)); print("JX belief std of luma coefficients (0..255 scale), step 2: DC %.1f, first AC %.1f, highest AC %.1f | luma quantisation steps: DC 160, highest 255" % (T["1"]["l"][0][0][0] ** 0.5, T["1"]["l"][0][0][1] ** 0.5, T["1"]["l"][0][7][7] ** 0.5), flush=True)
def launch(k, ex, test, sh, g0): return subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=f"jx{k}", KEY="jpeg", CFGSTR=f"jpeg {ex} {SCH}", NSTEP="4", MODES="zband8", CALSH="0:64 64:128", TESTDATA=test, TESTSH=sh, NGPU="8", PAR="2", GOFF=str(g0)), stdout=open(f"{L}/logs_x/launch_jx{k}.log", "w"), stderr=subprocess.STDOUT, cwd=L)
lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def sc(pat):
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{pat}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return (float(np.mean(Ps)), float(np.mean(Ls)), len(Ps)) if Ps else (float("nan"), float("nan"), 0)
ps = [launch(k, ex, f"{L}/data_gen512", "128:152 152:176", k) for k, ex in enumerate(V.values())]; [p.wait() for p in ps]; R = {}
for k, n in enumerate(V): R[n] = sc(f"jx{k}test_zband8_jpeg_jpeg"); print(f"JX generated images: {n}: {R[n][0]:.2f}/{R[n][1]:.3f} (n={R[n][2]})", flush=True)
b = max(R, key=lambda n: R[n][0] if R[n][2] == 48 else -1); kb = list(V).index(b); print("JX chosen on generated images:", b, flush=True)
subprocess.run(f"rm -rf {L}/exp_fin_jx*test_*", shell=True); launch("T", V[b], f"{L}/data_paper", "0:54 54:107", 0).wait(); r = sc("jxTtest_zband8_jpeg_jpeg")
print(f"JX TEST {b}: {r[0]:.2f}/{r[1]:.3f} (n={r[2]}) | current 27.19/0.336 | DAPS 28.15/0.259 | PnP-CM 26.05/0.358", flush=True)
