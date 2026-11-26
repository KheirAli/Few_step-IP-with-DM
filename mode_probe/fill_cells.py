"""Fill the empty cells of the two tables (run from PnP-CM/): PnP-CM and PnP-CM + scalar extrapolation on the remaining tasks, both datasets;
ours on bedroom for the new JPEG / blur-then-quantise steps and coded diffraction.  PnP-CM = the same loop with its own solver (or the exact linear
step for linear operators), no noise tilt, shipped schedule.  Weights always fitted on generated images of the same dataset."""
import subprocess, os, glob, re, time, sys, json, numpy as np, torch, lpips
from PIL import Image
L = "../mode_probe"; P = "../pnpcm_env/bin/python"; GP = os.environ.get("GPULIST", "0 1 2 3 7").split()
PNP = {"gamma": "deblur_nl TILT=0 NEWOP=gamma SOLVER=repo", "blurq": "deblur_nl TILT=0 NEWOP=blurq SOLVER=repo", "blurclip": "deblur_nl TILT=0 NEWOP=blurclip SOLVER=repo",
       "ct60": "deblur_nl TILT=0 NEWOP=ct CT_K=60 CT_SVD=1 RHOX=1", "ct30": "deblur_nl TILT=0 NEWOP=ct CT_K=30 CT_SVD=1 RHOX=1",
       "ctnl60": "deblur_nl TILT=0 NEWOP=ctnl CT_K=60 SOLVER=repo RHOX=1 REPO_LR=3e-2", "ctnl30": "deblur_nl TILT=0 NEWOP=ctnl CT_K=30 SOLVER=repo RHOX=1 REPO_LR=3e-2",
       "mri4": "deblur_nl TILT=0 NEWOP=mri MRI_R=4 RHOX=1", "mri8": "deblur_nl TILT=0 NEWOP=mri MRI_R=8 RHOX=1", "cdp2": "deblur_nl TILT=0 NEWOP=prcdp PR_MASKS=2 SOLVER=repo REPO_LR=3e-2",
       "pru": "deblur_nl TILT=0 NEWOP=prdaps PRD_K=0 SOLVER=repo"}
D = {"celeba": dict(gen=f"{L}/data_gen512", test=f"{L}/data_paper", sh="0:54 54:107", n=107, env={}), "bedroom": dict(gen=f"{L}/data_bedgen", test=f"{L}/data_bed", sh="0:29 29:58", n=58, env={"DSET": "bedroom", "MEANFACE": "mean_bed_gen.pt"})}
if not os.path.exists(f"{L}/mean_bed_gen.pt"):
    fs = sorted(glob.glob(f"{L}/data_bedgen/**/*.png", recursive=True))[:256]; torch.save(torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) for f in fs]).mean(0), f"{L}/mean_bed_gen.pt")
cb = json.load(open(f"{L}/clean_protocol_choicesbed.json")); sg = lambda c: f"SIGS={','.join(f'{v:.4f}' for v in c['sig'])} SIGPOS=0,1,2,3 SIGTOLD={c['told']}"; JB = os.path.realpath(f"{L}/jpeg_ev_table_bed.json")
J = []   # (dataset, tag, key, cfg, mode)
for t in ("gamma", "blurq", "blurclip", "ct60", "ct30", "ctnl60", "ctnl30", "mri4", "mri8", "cdp2", "pru"): J.append(("celeba", f"fcL{t}", t, PNP[t], "scalar"))
for t in ("ct60", "ct30", "pru"): J.append(("celeba", f"fcP{t}", t, PNP[t], "plain"))
for t in ("ct60", "ct30", "mri4", "mri8", "cdp2", "pru"): J.append(("bedroom", f"fbL{t}", t, PNP[t], "scalar"))
for t in ("ct60", "ct30", "cdp2", "pru"): J.append(("bedroom", f"fbP{t}", t, PNP[t], "plain"))
J += [("bedroom", "fbOjpeg", "jpeg", f"jpeg JPROJ=1 JCOLD=1 JEV=use JEVS=0.35 JEVFILE={JB} JFINAL=1 {sg(cb['jpeg'])}", "zband8"), ("bedroom", "fbOblurq", "blurq", f"deblur_nl NEWOP=blurq SPLIT_EV=0.3 {sg(cb['blurq'])}", "zband8"),
      ("bedroom", "fbOcdp2", "cdp2", "deblur_nl NEWOP=prcdp PR_MASKS=2 RHOX=0.3 PR_ISO_WARM=50 SIGS=0.1500,0.1375,0.1800,0.0900 SIGPOS=0,1,2,3 SIGTOLD=1.5", "zband8")]
if sys.argv[1:]: J = [j for j in J if j[1] in sys.argv[1:]]
if any(j[1] == "fbOjpeg" for j in J) and not os.path.exists(JB):     # belief widths for the JPEG posterior mean, measured on generated bedroom images
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("JCALGPU", GP[0]), DATA=os.path.realpath(D["bedroom"]["gen"]), SHARD="0:48", SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", TILT="0.5", JEV="cal", JEVFILE=JB, OMP_NUM_THREADS="4", DSET="bedroom")
    env.update(dict(kv.split("=", 1) for kv in f"JPROJ=1 JCOLD=1 {sg(cb['jpeg'])}".split())); subprocess.run([P, f"{L}/unified.py", "fin_fbjcal", "--deg=jpeg"], env=env, stdout=open(f"{L}/logs_x/fbjcal.out", "w"), stderr=subprocess.STDOUT); subprocess.run(f"rm -rf {L}/exp_fin_fbjcal_s0", shell=True)
run = []; g = 0
for ds, tag, key, cfg, mode in J:
    while len([p for p in run if p.poll() is None]) >= int(os.environ.get("MAXL", "5")): time.sleep(3)
    d = D[ds]
    if mode == "plain":
        c = cfg.split(); 
        for sh in d["sh"].split():
            env = dict(os.environ, GPULIST=" ".join(GP), DATA=os.path.realpath(d["test"]), SHARD=sh, SAMPLE_SEED="1", RESTART="never", NOISE="white", NOLOG="1", MODE="unified", OMP_NUM_THREADS="4", **d["env"]); env.update(dict(kv.split("=", 1) for kv in c[1:])); g += 1
            run.append(subprocess.Popen(["bash", f"{L}/gpurun.sh", f"{L}/logs_x/{tag}_{sh.split(':')[0]}.out", P, f"{L}/unified.py", f"fin_{tag}test_plain_{key}_{c[0]}", f"--deg={c[0]}"], env=env))
    else:
        env = dict(os.environ, TAG=tag, KEY=key, CFGSTR=cfg, NSTEP="4", MODES=mode, CALSH="0:64 64:128", TESTSH=d["sh"], CALDATA=d["gen"], TESTDATA=d["test"], NGPU="8", PAR="2", GOFF=str(g), GPULIST=" ".join(GP), **d["env"]); g += 2
        run.append(subprocess.Popen(["bash", f"{L}/launch_llewd.sh"], env=env, stdout=open(f"{L}/logs_x/launch_{tag}.log", "w"), stderr=subprocess.STDOUT, cwd=L))
[p.wait() for p in run]; lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
for ds, tag, key, cfg, mode in J:
    Ps = []; Ls = []
    for rf in glob.glob(f"{L}/exp_fin_{tag}test_{mode}_{key}_*_s*/raw_*.pt"):
        d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
        if key == "pru": c = [x, x.flip(-1, -2)]; x = c[int(np.argmin([float(((v - gt) ** 2).mean()) for v in c]))]          # unmasked phase retrieval: the 180-degree flip is not identifiable
        with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
    who = {"scalar": "PnP-CM + scalar extrapolation", "plain": "PnP-CM", "zband8": "Ours"}[mode]
    print(f"FILL | {ds} | {key} | {who} | {np.mean(Ps):.2f}/{np.mean(Ls):.3f}" + ("" if len(Ps) == D[ds]["n"] else f" (n={len(Ps)})") + " |" if Ps else f"FILL | {ds} | {key} | {who} | missing |", flush=True)
