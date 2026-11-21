import glob, re, json, sys, numpy as np, torch, lpips
from PIL import Image
dev=torch.device("cuda:0"); lp=lpips.LPIPS(net="vgg",verbose=False).to(dev)
for d0 in sorted(set(re.sub(r"_s\d+$","",d) for d in glob.glob(f"../mode_probe/exp_fin_mri{sys.argv[1]}_*_s*"))):
    P=[];L=[];T=[];R=[]
    for d in glob.glob(d0+"_s*"):
        S=json.load(open(d+"/stats.json")); T+=[x["time_total"] for x in S]; R+=[x["ref"] for x in S]
        for rf in glob.glob(d+"/raw_*.pt"):
            i=int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab=glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab: continue
            gt=torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32)/255).permute(2,0,1)*2-1; x=torch.load(rf)["x_raw"][0].float()
            with torch.no_grad(): P.append(float(10*torch.log10(1/(((x-gt)/2)**2).mean()))); L.append(float(lp(x[None].to(dev),gt[None].to(dev))))
    print(f"MRI {d0.split('exp_fin_mri')[1]:22s} n={len(P):3d} zero-filled {np.mean(R):.2f} dB -> {np.mean(P):.2f}/{np.mean(L):.3f}  time {np.median(T):.2f}s", flush=True)
