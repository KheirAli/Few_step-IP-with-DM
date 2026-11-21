import glob, re, json, sys, numpy as np, torch, lpips
from PIL import Image
dev=torch.device("cuda:0"); lp=lpips.LPIPS(net="vgg",verbose=False).to(dev)
T=[("sr_bicubic","sr_bicubic"),("deblur_gauss","deblur_gauss"),("inpainting","inpainting"),("jpeg","jpeg"),("deblur_nl","deblur_nl"),("hdr","deblur_nl"),("quant","deblur_nl"),("onebit","deblur_nl")]
def score(pat):
    P=[];L=[];tm=[]
    for d in glob.glob(f"../mode_probe/exp_fin_{pat}_s*"):
        if glob.glob(d+"/stats.json"): tm+=[x["time_total"] for x in json.load(open(d+"/stats.json")) if "time_total" in x]
        for rf in glob.glob(d+"/raw_*.pt"):
            i=int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab=glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
            if not lab: continue
            gt=torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32)/255).permute(2,0,1)*2-1; x=torch.load(rf)["x_raw"][0].float()
            with torch.no_grad(): P.append(float(10*torch.log10(1/(((x-gt)/2)**2).mean()))); L.append(float(lp(x[None].to(dev),gt[None].to(dev))))
    return (np.mean(P),np.mean(L),len(P),np.median(tm) if tm else float("nan")) if P else None
mode=sys.argv[1]
if mode=="val":
    for t,d in T:
        r=score(f"pptest_ship4_{t}_{d}")
        if r: print(f"VALIDATION {t:12s} PnP-CM solver in our loop, shipped 4 steps: {r[0]:.2f}/{r[1]:.3f} n={r[2]} time {r[3]:.2f}s")
if mode=="sel":
    best={}
    for t,d in T:
        rows=[(n,score(f"ppsel_{n}_{t}_{d}")) for n in ["p0_1","p0_2","p0_3","p1_2","p1_3","p05_2","p05_25","p0_15"]]; rows=[(n,r) for n,r in rows if r]
        if rows: best[t]=max(rows,key=lambda q:q[1][0])[0]; print(f"SEL {t:12s} "+" | ".join(f"{n} {r[0]:.2f}/{r[1]:.3f}" for n,r in rows)+f" -> {best[t]}")
    json.dump(best, open("../mode_probe/pp2_best.json","w"))
if mode=="test":
    for t,d in T:
        a=score(f"pptest_2ours_{t}_{d}"); b=score(f"pptest_2own_{t}_{d}")
        if a and b: print(f"TEST {t:12s} PnP-CM solver, 2 steps: on our schedule {a[0]:.2f}/{a[1]:.3f} ({a[3]:.2f}s) | on its own best schedule {b[0]:.2f}/{b[1]:.3f} ({b[3]:.2f}s) n={a[2]}")
