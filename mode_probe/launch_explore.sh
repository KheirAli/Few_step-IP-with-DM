#!/bin/bash
# Targeted exploration (2026-11-17): unified solver, white noise, restart rule r2, 4 sampler seeds, 100 images, all tasks.
#   xw = noise weighted by the uncertainty map (same budget)   xg = 1.5x noise everywhere (control)   xr = regional high-noise resample
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe
units=()
for seed in 1 2 3 4; do for x in w g r; do for sh in 0:25 25:50 50:75 75:100; do units+=("$x $seed $sh"); done; done; done
lane(){ l=$1; gpu=$((l % 8)); for ((i=l; i<${#units[@]}; i+=16)); do set -- ${units[$i]}; x=$1; seed=$2; sh=$3
  for deg in sr_bicubic deblur_gauss inpainting jpeg deblur_nl phase_retrieval; do
    env CUDA_VISIBLE_DEVICES=$gpu DATA=$D SHARD=$sh SAMPLE_SEED=$seed MODE=unified RESTART=rule RULE=r2 NOISE=white EXPLORE=$x $P $L/unified.py fin_x${x}${seed}_$deg --deg=$deg > $L/logs_x/x${x}${seed}_${deg}_s${sh%%:*}.out 2>&1
  done; done; }
for l in $(seq 0 15); do lane $l & done; wait; echo ALLDONE
