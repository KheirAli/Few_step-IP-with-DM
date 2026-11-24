#!/bin/bash
# Overnight queue (2026-11-24).  Phases run one after another so that the GPUs are never over-committed; each writes its lines to overnight/<phase>.txt.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; O=$L/overnight; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
T=/tmp/claude-3334177/-usr-scratch-akheirandish3/2dbff445-c579-4445-b75d-c211275efedc/tasks
echo "start $(date)" > $O/status.txt
for i in $(seq 1 360); do a=$(grep -c "A2 TEST masked" $T/bdqc1hvr0.output 2>/dev/null); b=$(grep -c "PRT TEST learned predictor, 8 calls" $T/bv64646df.output 2>/dev/null); [ "${a:-0}" -ge 1 ] && [ "${b:-0}" -ge 1 ] && break; sleep 30; done
grep -E "^A2" $T/bdqc1hvr0.output > $O/0_attempt2.txt 2>/dev/null; grep -E "^PRT" $T/bv64646df.output > $O/0_pr_today.txt 2>/dev/null; echo "phase 0 done $(date)" >> $O/status.txt
ph() { local name=$1; shift; echo "$name start $(date)" >> $O/status.txt; "$@" 2>&1 | grep -E "^P2|^\||^CHOICE|^MRIB|Error|Traceback" > $O/$name.txt; echo "$name done $(date)" >> $O/status.txt; }
ph 1_masked_pr_attempt2 $P $L/pr2fw.py
ph 2_new_modes $P $L/new_modes.py
ph 3_mri_celeba env PREFIX=m TASKS="mri4 mri8" WAVE=8 $P $L/clean_protocol.py
ph 4_bedroom_clean env DSET=bedroom PREFIX=bed GENDATA=$L/data_bedgen CALDATA=$L/data_bedgen TESTDATA=$L/data_bed TESTSH="0:29 29:58" NTEST=58 WAVE=12 TASKS="sr_bicubic deblur_gauss inpainting jpeg deblur_nl hdr quant onebit blurclip blurq gamma ct30 ct60 mri4 mri8" $P $L/clean_protocol.py
# PnP-CM on the MRI-type operator (its linear step is the same exact formula; no tilt, no combination), both datasets
echo "5_mri_pnp start $(date)" >> $O/status.txt; g=0
for ds in "celeba|$L/data_paper|0:54 54:107" "bedroom|$L/data_bed|0:29 29:58"; do IFS='|' read dn dd shs <<< "$ds"; for R in 4 8; do for sh in $shs; do
  env DSET=$dn CUDA_VISIBLE_DEVICES=$((g%8)) OMP_NUM_THREADS=4 DATA=$(realpath $dd) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0 NEWOP=mri MRI_R=$R RHOX=1 $P $L/unified.py fin_mripnp_${dn}_r$R --deg=deblur_nl > $L/logs_x/mripnp_${dn}_r${R}_${sh%%:*}.out 2>&1 & g=$((g+1)); done; done; done; wait
CUDA_VISIBLE_DEVICES=0 $P - > $O/5_mri_pnp.txt 2>&1 <<'PY'
import glob, re, os, numpy as np, torch, lpips
from PIL import Image
lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
for dn in ("celeba", "bedroom"):
    for R in (4, 8):
        Ps = []; Ls = []
        for rf in glob.glob(f"../mode_probe/exp_fin_mripnp_{dn}_r{R}_s*/raw_*.pt"):
            d = os.path.dirname(rf); i = int(re.findall(r"raw_(\d+)\.pt", rf)[0]); lab = glob.glob(f"{d}/image_samples/*/labels/orig_{i}.png"); gt = torch.from_numpy(np.asarray(Image.open(lab[0]).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1; x = torch.load(rf)["x_raw"][0].float().clamp(-1, 1)
            with torch.no_grad(): Ps.append(float(10 * torch.log10(1 / (((x - gt) / 2) ** 2).mean()))); Ls.append(float(lp(x[None].cuda(), gt[None].cuda())))
        print(f"MRIB PnP-CM, {dn}, MRI-type sampling R={R}: {np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)})" if Ps else f"MRIB {dn} R={R} missing")
PY
echo "5_mri_pnp done $(date)" >> $O/status.txt
ph 6_timing env TGPU=0 $P $L/timing_pass.py
echo "ALL DONE $(date)" >> $O/status.txt
