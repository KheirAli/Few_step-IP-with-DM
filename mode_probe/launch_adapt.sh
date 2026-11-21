#!/bin/bash
# Diagnostics + the three adaptive mechanisms, 4 model calls, 24 selection images (SET=sel) or the 107 test images (SET=test).
#   diag : current method with per-step diagnostics            disc : noise-calibrated automatic penalty (linear tasks)
#   damp : measurement-preserving damping of the model output   null : noise only in weakly measured directions (linear tasks)
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; SET=${SET:-sel}; g=0
if [ $SET = sel ]; then D=$(realpath $L/data100); SH="0:24"; else D=$(realpath $L/data_paper); SH="0:54 54:107"; fi
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" )
LIN="sr_bicubic deblur_gauss inpainting"
declare -A VAR=( [diag]="DIAG2=1" [disc1]="DISC=1.0 DIAG2=1" [disc15]="DISC=1.5" [disc07]="DISC=0.7" [damp1]="DAMP=1.0 DIAG2=1" [damp15]="DAMP=1.5" [damp2]="DAMP=2.0" [damp3]="DAMP=3.0" [damplast]="DAMP=1.5 DAMPSTEPS=3"
  [nullhard]="NULLNOISE=hard" [nullkeep]="NULLNOISE=keep" [discdamp]="DISC=1.0 DAMP=1.5" )
for task in ${TASKS:-sr_bicubic deblur_gauss inpainting jpegtrue deblur_nl hdr quant}; do c=(${CFG[$task]}); deg=${c[0]}
  for v in ${VARS:-diag disc1 disc15 disc07 damp1 damp15 damp2 damp3 damplast nullhard nullkeep discdamp}; do
    if [[ $v == disc* || $v == null* ]] && [[ ! " $LIN " =~ " $task " ]]; then continue; fi
    for sh in $SH; do env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} ${VAR[$v]} $P $L/unified.py fin_ad${SET}_${v}_${task}_$deg --deg=$deg > $L/logs_x/ad${SET}_${v}_${task}_${sh%%:*}.out 2>&1 &
      g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; done; done; done; wait; echo ALLDONE
