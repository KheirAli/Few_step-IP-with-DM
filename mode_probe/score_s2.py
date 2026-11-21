import glob, itertools, re, json, sys, numpy as np, torch, lpips
from PIL import Image
dev=torch.device("cuda:0"); lp=lpips.LPIPS(net="vgg",verbose=False).to(dev); F=torch.tensor(list(itertools.product([1.0,-1.0],repeat=3))).view(8,3,1,1)
TASKS=[("sr_bicubic","sr_bicubic"),("deblur_gauss","deblur_gauss"),("inpainting","inpainting"),("jpeg","jpeg"),("jpegtrue","jpeg"),("deblur_nl","deblur_nl"),("prmask","phase_retrieval"),("hdr","deblur_nl"),("quant","deblur_nl"),("onebit","deblur_nl")]
def score(pat, pr):
    R={}; T=[]; N=[]
    for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
        if glob.glob(d+"/stats.json"): S=json.load(open(d+"/stats.json")); T+=[x["time_total"] for x in S if "time_total" in x]; N+=[x["nfe"] for x in S]
        for rf in glob.glob(d+"/raw_*.pt"):
            i=int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab=glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab: continue
            gt=torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32)/255).permute(2,0,1)*2-1; x=torch.load(rf)["x_raw"][0].float()
            if pr: c=F*x; x=c[int(((c-gt)**2).mean(dim=(1,2,3)).argmin())]
            with torch.no_grad(): R[i]=(float(10*torch.log10(1/(((x-gt)/2)**2).mean())), float(lp(x[None].to(dev),gt[None].to(dev))))
    return R, (np.median(T) if T else float("nan")), (np.mean(N) if N else 0)
SET=sys.argv[1]; names=["p0_1","p0_2","p0_3","p1_2","p1_3","p05_2","p05_25","p0_15"]; best={}
for task,deg in TASKS:
    rows=[]
    for n in names:
        R,t,nf=score(f"s2{SET}_{task}_{n}_{deg}", task=="prmask")
        if R: a=np.array(list(R.values())); rows.append((n,a[:,0].mean(),a[:,1].mean(),t,nf,len(R)))
    if not rows: continue
    if SET=="sel": best[task]=max(rows,key=lambda r:r[1])[0]
    print(f"{task:13s} "+" | ".join(f"{n} {p:.2f}/{l:.3f}" for n,p,l,t,nf,k in rows)+(f"  -> {best[task]}" if SET=="sel" else f"  (calls {rows[0][4]:.0f}, {rows[0][3]:.2f}s, n={rows[0][5]})"), flush=True)
if SET=="sel": json.dump(best, open("../mode_probe/s2_best.json","w"))
