#!/bin/bash
# Random flip / shift around every model call, on the 24 selection images (final 4-call settings).
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4; P=../pnpcm_env/bin/python; L=../mode_probe; GL=(0 1 2 4 5 7); g=0
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [inpainting]="inpainting LINEXACT=1 FW=1 FWP=0.5" [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1 FW=1 FWP=0.5 FWSTEPS=1,2,3" [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant FW=1 FWP=0.5 FWMIN=0.4" [onebit]="deblur_nl NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2 FW=1 FWP=0.5 FWMIN=0.4" )
declare -A W=( [sr_bicubic]=sr_bicubic [inpainting]=best_inpainting [jpeg]=best_jpeg [hdr]=hdr [quant]=best_quant [onebit]=best_onebit )
for task in sr_bicubic inpainting jpeg hdr quant onebit; do c=(${CFG[$task]}); for v in "base|" "flip|EQ=flip" "shift4|EQ=shift4" "shift16|EQ=shift16" "flipshift8|EQ=flipshift8"; do for lw in lle nolle; do
  [ $lw = lle ] && X="LLE=$(realpath $L/lle)/${W[$task]}.json" || X=""
  env CUDA_VISIBLE_DEVICES=${GL[$((g%6))]} DATA=$(realpath $L/data100) SHARD=0:24 SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} ${v#*|} $X $P $L/unified.py fin_eq_${v%%|*}_${lw}_${task} --deg=${c[0]} > $L/logs_x/eq_${v%%|*}_${lw}_${task}.out 2>&1 & g=$((g+1)); if (( g % 18 == 0 )); then wait; fi; done; done; done; wait; echo ALLDONE
