#!/bin/bash
# PnP-CM's gradient solver on the CT operator: learning rate chosen on 24 non-test images (shipped value and two smaller), then the 107 test images.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; export OMP_NUM_THREADS=4; g=0
C="SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified NEWOP=ct TILT=0 SOLVER=repo"
for lr in 5e-4 1e-4 2e-5 5e-6; do for sc in "ours|RHOX=3 SIGS=0.5,0.25,0.1,0.02" "ship|RHOX=1"; do
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $L/data100) SHARD=0:24 $C REPO_LR=$lr ${sc#*|} $P $L/unified.py fin_ctsel_gd_${sc%%|*}_lr$lr --deg=deblur_nl > $L/logs_x/ctsel_gd_$g.out 2>&1 & g=$((g+1)); done; done; wait
best=$($P - <<'EOF'
import json, glob, re, numpy as np, torch
from PIL import Image
r={}
for d in glob.glob("../mode_probe/exp_fin_ctsel_gd_*_s0"):
    P=[]
    for rf in glob.glob(d+"/raw_*.pt"):
        i=int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab=glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png")
        gt=torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32)/255).permute(2,0,1)*2-1; x=torch.load(rf)["x_raw"][0].float(); P.append(float(10*torch.log10(1/(((x-gt)/2)**2).mean())))
    if P: r[d.split("ctsel_gd_")[1].split("_s0")[0]]=np.mean(P)
import sys
for k,v in sorted(r.items(), key=lambda kv:-kv[1]): print(f"GDSEL {k}: {v:.2f} dB", file=sys.stderr)
k=max(r,key=r.get); sc,lr=k.split("_lr"); print(("RHOX=3 SIGS=0.5,0.25,0.1,0.02" if sc=="ours" else "RHOX=1")+f" REPO_LR={lr}")
EOF
); echo "GD chosen: $best"
for K in 60 30; do for sh in 0:54 54:107; do env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $L/data_paper) SHARD=$sh $C CT_K=$K $best $P $L/unified.py fin_cttest_k${K}_pnpgd --deg=deblur_nl > $L/logs_x/cttest_k${K}_pnpgd_${sh%%:*}.out 2>&1 & g=$((g+1)); done; done; wait; echo ALLDONE
