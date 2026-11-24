"""Round 3 (run from PnP-CM/).  Everything tuned on MODEL-GENERATED images, test set used once per configuration.
1. blur then clip: three data-fits (8-pass split, expected-value split, ACTIVE SET) - generated and test, with the true data-fit objective.
2. nonlinear CT with NO alternation: penalty x coupling weight x noise schedule tuned (combination in the objective), weights refitted, test.
3. phase retrieval where the transform is an isometry: coded diffraction with 2 phase masks (tuned, vs PnP-CM's solver), unpadded Fourier."""
import json, subprocess, os, glob, re, time, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; lp = None
def launch(tag, cfg, data, shard, g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % 8), DATA=os.path.realpath(data), SHARD=shard, SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", OMP_NUM_THREADS="4")
    env.update(dict(kv.split("=", 1) for kv in cfg.split()))
    return subprocess.Popen([P, f"{L}/unified.py", f"fin_{tag}", "--deg=deblur_nl"], env=env, stdout=open(f"{L}/logs_x/{tag}_{shard.split(':')[0]}.out", "w"), stderr=subprocess.STDOUT)
def score(tag, with_lpips=True):
    global lp; Ps = []; Ls = []; J = []
    for d in glob.glob(f"{L}/exp_fin_{tag}_s*"):
        if os.path.exists(d + "/stats.json"): J += [np.mean(s["bcJ"], 0)[1] for s in json.load(open(d + "/stats.json")) if "bcJ" in s]
        for rf in glob.glob(d + "/raw_*.pt"):
            i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
            Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean())))
            if with_lpips:
                if lp is None: lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
                with torch.no_grad(): Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    return f"{np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)})" + (f", objective {np.mean(J):.0f}" if J else "") if Ps else "missing", (np.mean(Ps) if Ps else float("nan"))
# ---- stage A: everything that needs no schedule search, started at once
BC = {"split8": "NEWOP=blurclip BC_JLOG=1", "ev0.3": "NEWOP=blurclip SPLIT_EV=0.3 BC_JLOG=1", "active": "NEWOP=blurclip BC_MODE=active BC_JLOG=1"}
jobs = [launch(f"r3bc_{n}_{w}", c, f"{L}/data_gen512" if w == "gen" else f"{L}/data_paper", sh, g) for g, (n, c, w, sh) in enumerate([(n, c, w, sh) for n, c in BC.items() for w, shs in (("gen", ["0:48"]), ("test", ["0:54", "54:107"])) for sh in shs])]
jobs += [launch("r3sq", "NEWOP=prsq", f"{L}/data100", "0:24", 5), launch("r3sqw", "NEWOP=prsq PR_ISO_WARM=50", f"{L}/data100", "0:24", 6)]
PN = {"def": "", "1e-2": "REPO_LR=1e-2", "3e-2": "REPO_LR=3e-2", "1e-1": "REPO_LR=1e-1"}
jobs += [launch(f"r3cdpnp_{n}_gen", f"NEWOP=prcdp PR_MASKS=2 SOLVER=repo TILT=0 {c}", f"{L}/data_gen512", "0:48", g) for g, (n, c) in enumerate(PN.items())]
# ---- stage B: schedule searches (one pass of coordinate search with the combination in the objective)
keys = [f"ctnl30o_r{r}_m{m}" for r in ("0.03", "0.1", "0.3") for m in ("1", "2", "4")] + [f"cdp_r{r}_w{w}" for r in ("0.3", "1", "3") for w in ("0", "50")]
sp = [subprocess.Popen([P, f"{L}/sched_cd2.py", k, str(g)], env=dict(os.environ, NSTEPS="4"), stdout=open(f"{L}/logs_x/r3_cd2_{k}.log", "w"), stderr=subprocess.STDOUT) for g, k in enumerate(keys)]
[p.wait() for p in jobs]
print("BLURCLIP generated: " + " | ".join(f"{n}: {score(f'r3bc_{n}_gen')[0]}" for n in BC), flush=True)
print("BLURCLIP test:      " + " | ".join(f"{n}: {score(f'r3bc_{n}_test')[0]}" for n in BC), flush=True)
print(f"UNPADDED Fourier magnitudes, 24 selection images, mean-face start: {score('r3sq')[0]} | with 50 classical magnitude-fit passes first: {score('r3sqw')[0]}", flush=True)
pn = {n: score(f"r3cdpnp_{n}_gen", False)[1] for n in PN}; pbest = max(pn, key=lambda k: pn[k] if pn[k] == pn[k] else -1); print("CDP PnP-CM solver on generated images: " + " | ".join(f"{n}: {v:.2f}" for n, v in pn.items()) + f" || chosen {pbest}", flush=True)
[p.wait() for p in sp]
R = {k: json.load(open(f"{L}/sched_cd2_{k}.json"))["4"] for k in keys if os.path.exists(f"{L}/sched_cd2_{k}.json")}
for name, pref in (("NONLINEAR CT 30 views, no alternation", "ctnl30o"), ("CODED DIFFRACTION 2 masks", "cdp")):
    print(f"TUNE {name}: " + " | ".join(f"{k.split('_', 1)[1]}: {v['start'][0]:.2f}->{v['best']:.2f}" for k, v in R.items() if k.startswith(pref)), flush=True)
bct = max([k for k in R if k.startswith("ctnl30o")], key=lambda k: R[k]["best"]); bcd = max([k for k in R if k.startswith("cdp")], key=lambda k: R[k]["best"])
src = open(f"{L}/sched_search.py").read(); exec(src[src.index("CFG = {"):src.index("deg, extra, base")])
def cfgstr(key, extra=None, told=None):
    r = R[key]; return f"deblur_nl {extra or CFG[key][1]} SIGS={','.join(f'{v:.4f}' for v in r['sig'])} SIGPOS=0,1,2,3 SIGTOLD={r['told']}"
fin = [("ctnlO30", "ctnl30", cfgstr(bct)), ("ctnlO60", "ctnl60", cfgstr(bct, CFG[bct][1].replace("CT_K=30", "CT_K=60"))), ("cdpF", "cdp2", cfgstr(bcd))]
print(f"CHOSEN nonlinear CT: {bct} sigma {' '.join(f'{v:.3f}' for v in R[bct]['sig'])} told x{R[bct]['told']} | coded diffraction: {bcd} sigma {' '.join(f'{v:.3f}' for v in R[bcd]['sig'])} told x{R[bcd]['told']}", flush=True)
fp = [subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=dict(os.environ, TAG=t, KEY=k, CFGSTR=c, NSTEP="4", MODES="scalar zband8", CALSH="0:64 64:128", NGPU="8", PAR="4", GOFF=str(3 * g)), stdout=open(f"{L}/logs_x/launch_{t}.log", "w"), stderr=subprocess.STDOUT, cwd=L) for g, (t, k, c) in enumerate(fin)]
fp += [launch("r3cdpnp_test", f"NEWOP=prcdp PR_MASKS=2 SOLVER=repo TILT=0 {PN[pbest]}", f"{L}/data_paper", sh, 6 + g) for g, sh in enumerate(("0:54", "54:107"))]
[p.wait() for p in fp]
for t, k, _ in fin:
    print(f"FINAL {t}: " + " | ".join(f"{m}: {score(f'{t}test_{m}_{k}_deblur_nl')[0]}" for m in ("scalar", "zband8")), flush=True)
print(f"FINAL coded diffraction, PnP-CM solver ({pbest}): {score('r3cdpnp_test')[0]}", flush=True)
