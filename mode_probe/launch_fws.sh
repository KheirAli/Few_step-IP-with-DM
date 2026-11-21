#!/bin/bash
# Frequency weighting with the first step excluded (FWSTEPS=1,2,3), both strengths, all tasks, 24 selection images, 4 steps.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; export OMP_NUM_THREADS=4; g=0
while pgrep -f "unified.py fin_fws[0-3]_" > /dev/null; do sleep 20; done
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1" [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit" )
for task in sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit; do c=(${CFG[$task]}); deg=${c[0]}
  for v in "a|FWP=0.5 FWSTEPS=1,2,3" "b|FWP=0.5 FWMIN=0.4 FWSTEPS=1,2,3" "c|FWP=0.5 FWSTEPS=2,3"; do
    env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=0:24 SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} FW=1 ${v#*|} $P $L/unified.py fin_fwx${v%%|*}_${task}_$deg --deg=$deg > $L/logs_x/fwx${v%%|*}_$task.out 2>&1 & g=$((g+1)); if (( g % 8 == 0 )); then wait; fi; done; done; wait; echo ALLDONE
