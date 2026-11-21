#!/bin/bash
# Parallel-samples study: the shipped 4-step method with sampler seeds 2,3,4 on the 107 test images (seed 1 = fin_s8test_*_ship4). Same measurements.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1"
  [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit" )
g=0
for task in sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit; do set -- ${CFG[$task]}; deg=$1; shift
  for sd in 2 3 4; do for sh in 0:54 54:107; do
    env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=$sd RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 "$@" $P $L/unified.py fin_par${sd}_${task}_$deg --deg=$deg > $L/logs_x/par${sd}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1))
    if (( g % 16 == 0 )); then wait; fi; done; done
done; wait; echo ALLDONE
