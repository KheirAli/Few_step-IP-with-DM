"""JPEG follow-up: belief-width factor and schedule for the posterior-mean step, chosen on generated images; then the test set once (run from PnP-CM/)."""
import json, subprocess, os, glob, re, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; c = json.load(open(f"{L}/clean_protocol_choices.json"))["jpeg"]; JF = os.path.realpath(f"{L}/jpeg_ev_table.json"); B = "JPROJ=1 JCOLD=1"
def sch(s, t): return f"SIGS={','.join(f'{v:.4f}' for v in s)} SIGPOS=0,1,2,3 SIGTOLD={t}"
def launch(k, ex, test, sh, g0): return subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=f"jy{k}", KEY="jpeg", CFGSTR=f"jpeg {ex}", NSTEP="4", MODES="zband8", CALSH="0:64 64:128", TESTDATA=test, TESTSH=sh, NGPU="8", PAR="2", GOFF=str(g0)), stdout=open(f"{L}/logs_x/launch_jy{k}.log", "w"), stderr=subprocess.STDOUT, cwd=L)
lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
def sc(pat):
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{pat}_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return (float(np.mean(Ps)), float(np.mean(Ls)), len(Ps)) if Ps else (float("nan"), float("nan"), 0)
W = ["0.15", "0.25", "0.35", "0.5", "0.7"]; ps = [launch(f"w{k}", f"{B} JEV=use JEVS={w} JEVFILE={JF} JFINAL=1 {sch(c['sig'], c['told'])}", f"{L}/data_gen512", "128:152 152:176", k) for k, w in enumerate(W)]; [p.wait() for p in ps]
R = {w: sc(f"jyw{k}test_zband8_jpeg_jpeg") for k, w in enumerate(W)}; print("JY width factor on generated images: " + " | ".join(f"x{w}: {r[0]:.2f}/{r[1]:.3f}" for w, r in R.items()), flush=True); wb = max(R, key=lambda w: R[w][0] if R[w][2] == 48 else -1)
ex = json.load(open(f"{L}/sched_cfg_extra.json")); ex["jpeg__ev"] = ["jpeg", f"{B} JEV=use JEVS={wb} JEVFILE={JF}", [float(v) for v in c["sig"]]]; json.dump(ex, open(f"{L}/sched_cfg_extra.json", "w"))
subprocess.run([P, f"{L}/sched_cd2.py", "jpeg__ev", "0"], env=dict(os.environ, NSTEPS="4"), stdout=open(f"{L}/logs_x/jy_cd2.log", "w"), stderr=subprocess.STDOUT); r = json.load(open(f"{L}/sched_cd2_jpeg__ev.json"))["4"]
print(f"JY chosen width x{wb}; schedule tuning on generated images: {r['start'][0]:.2f} -> {r['best']:.2f}, sigma " + " ".join(f"{v:.3f}" for v in r["sig"]) + f" told x{r['told']}", flush=True)
subprocess.run(f"rm -rf {L}/exp_fin_jy*test_*", shell=True); cf = f"{B} JEV=use JEVS={wb} JEVFILE={JF} JFINAL=1 "
a = launch("G", cf + sch(r["sig"], r["told"]), f"{L}/data_gen512", "128:152 152:176", 0); a.wait(); g = sc("jyGtest_zband8_jpeg_jpeg"); print(f"JY generated images, tuned schedule: {g[0]:.2f}/{g[1]:.3f} (old schedule {R[wb][0]:.2f}/{R[wb][1]:.3f})", flush=True)
use = (r["sig"], r["told"]) if g[0] > R[wb][0] else (c["sig"], c["told"]); subprocess.run(f"rm -rf {L}/exp_fin_jy*test_*", shell=True)
launch("T", cf + sch(*use), f"{L}/data_paper", "0:54 54:107", 0).wait(); t = sc("jyTtest_zband8_jpeg_jpeg")
print(f"JY TEST posterior mean x{wb}, {'tuned' if use[0] is r['sig'] else 'old'} schedule: {t[0]:.2f}/{t[1]:.3f} (n={t[2]}) | first exact version 27.40/0.324 | previous 27.19/0.336 | DAPS 28.15/0.259", flush=True)
