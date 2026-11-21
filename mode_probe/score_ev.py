import glob, itertools, re, json, sys, numpy as np, torch, lpips
from PIL import Image
dev=torch.device("cuda:0"); lp=lpips.LPIPS(net="vgg",verbose=False).to(dev); F=torch.tensor(list(itertools.product([1.0,-1.0],repeat=3))).view(8,3,1,1); best=json.load(open("../mode_probe/s2_best.json"))
TASKS=[("sr_bicubic","sr_bicubic"),("deblur_gauss","deblur_gauss"),("inpainting","inpainting"),("jpeg","jpeg"),("jpegtrue","jpeg"),("deblur_nl","deblur_nl"),("prmask","phase_retrieval"),("hdr","deblur_nl"),("quant","deblur_nl"),("onebit","deblur_nl")]
def score(pat, pr):
    R={}
    for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
        for rf in glob.glob(d+"/raw_*.pt"):
            i=int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab=glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab: continue
            gt=torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32)/255).permute(2,0,1)*2-1; x=torch.load(rf)["x_raw"][0].float()
            if pr: c=F*x; x=c[int(((c-gt)**2).mean(dim=(1,2,3)).argmin())]
            with torch.no_grad(): R[i]=(float(10*torch.log10(1/(((x-gt)/2)**2).mean())), float(lp(x[None].to(dev),gt[None].to(dev))))
    return R
SET=sys.argv[1]; out={}
for st in (4,2):
    print(f"===== {SET}, {st} steps: base | hand-set weighting | calibrated variants (change in dB vs base) =====")
    for task,deg in TASKS:
        if SET=="sel": bp=f"fxsel_base_{task}_{deg}" if st==4 else f"s2sel_{task}_{best[task]}_{deg}"; hp=[f"fxsel_fw05_{task}_{deg}",f"fxsel_fw05m_{task}_{deg}"] if st==4 else [f"s2selfw05_{task}_{best[task]}_{deg}",f"s2selfw05m_{task}_{best[task]}_{deg}"]
        else: bp=f"fxtest_base_{task}_{deg}" if st==4 else f"s2test_{task}_{best[task]}_{deg}"; hp=[f"fxtest_fw05_{task}_{deg}",f"fxtest_fw05m_{task}_{deg}"] if st==4 else [f"s2testfw05_{task}_{best[task]}_{deg}",f"s2testfw05m_{task}_{best[task]}_{deg}"]
        B=score(bp, task=="prmask")
        if not B: continue
        b=np.array([B[i] for i in sorted(B)]); line=f"{task:13s} base {b[:,0].mean():.2f}/{b[:,1].mean():.3f}"
        hb=[np.array(list(score(h, task=="prmask").values())) for h in hp]; hb=[h for h in hb if len(h)]
        if hb: h=max(hb,key=lambda a:a[:,0].mean()); line+=f" | hand {h[:,0].mean():.2f}/{h[:,1].mean():.3f} ({h[:,0].mean()-b[:,0].mean():+.2f})"
        for v in ("fwcal","fwcal2","fwfull","ev","evfw"):
            R=score(f"ev{SET}{st}_{v}_{task}_{deg}", task=="prmask")
            if R: r=np.array(list(R.values())); out[f"{st}|{task}|{v}"]=[float(r[:,0].mean()),float(r[:,1].mean()),float(b[:,0].mean()),float(b[:,1].mean()),len(R)]; line+=f" | {v} {r[:,0].mean():.2f}/{r[:,1].mean():.3f} ({r[:,0].mean()-b[:,0].mean():+.2f})"
        print(line, flush=True)
json.dump(out, open(f"../mode_probe/ev_scores_{SET}.json","w"))
