import glob, itertools, re, json, sys, numpy as np, torch, lpips
from PIL import Image
dev=torch.device("cuda:0"); lp=lpips.LPIPS(net="vgg",verbose=False).to(dev); F=torch.tensor(list(itertools.product([1.0,-1.0],repeat=3))).view(8,3,1,1)
TASKS=[("sr_bicubic","sr_bicubic"),("deblur_gauss","deblur_gauss"),("inpainting","inpainting"),("jpeg","jpeg"),("jpegtrue","jpeg"),("deblur_nl","deblur_nl"),("prmask","phase_retrieval"),("hdr","deblur_nl"),("quant","deblur_nl"),("onebit","deblur_nl")]
def score(pat, pr):
    R={}; T=[]
    for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
        if glob.glob(d+"/stats.json"): T+=[x["time_total"] for x in json.load(open(d+"/stats.json")) if "time_total" in x]
        for rf in glob.glob(d+"/raw_*.pt"):
            i=int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab=glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab: continue
            gt=torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32)/255).permute(2,0,1)*2-1; x=torch.load(rf)["x_raw"][0].float()
            if pr: c=F*x; x=c[int(((c-gt)**2).mean(dim=(1,2,3)).argmin())]
            with torch.no_grad(): R[i]=(float(10*torch.log10(1/(((x-gt)/2)**2).mean())), float(lp(x[None].to(dev),gt[None].to(dev))))
    return R, (np.median(T) if T else float("nan"))
out={}
for SET in sys.argv[1:2]:
    print(f"===== {SET}: PSNR / LPIPS (change vs base, paired) | time per image =====")
    for task,deg in TASKS:
        B,tb=score(f"fx{SET}_base_{task}_{deg}", task=="prmask")
        if not B: continue
        b=np.array([B[i] for i in sorted(B)]); line=f"{task:13s} n={len(B):3d} base {b[:,0].mean():.2f}/{b[:,1].mean():.3f} ({tb:.2f}s)"
        for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("fw05","fw10","ft15","ft20","lt1")):
            R,t=score(f"fx{SET}_{v}_{task}_{deg}", task=="prmask"); ids=sorted(set(R)&set(B))
            if not ids: continue
            r=np.array([R[i] for i in ids]); bb=np.array([B[i] for i in ids]); out[f"{SET}|{task}|{v}"]=[float(r[:,0].mean()-bb[:,0].mean()), float(r[:,1].mean()-bb[:,1].mean()), float(np.mean(r[:,0]>bb[:,0])), len(ids)]
            line+=f" | {v} {r[:,0].mean():.2f}/{r[:,1].mean():.3f} ({r[:,0].mean()-bb[:,0].mean():+.2f}, {r[:,1].mean()-bb[:,1].mean():+.3f}; {t:.2f}s)"
        print(line, flush=True)
json.dump(out, open("../mode_probe/fix_scores.json","w"))
