#!/bin/bash
# High-frequency-tilted Gaussian noise (2026-11-17): same total noise power, same network calls as the baseline, 100 images, seed 1.
#   tw<a> = white budget tilted by exponent a     ts1 = shaped table with tilt 1
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; G=(${GPUS:-0 1 2 3 7}); n=${#G[@]}
jobs_=()
for spec in "twm1 white -1" "tw05 white 0.5" "tw1 white 1" "tw2 white 2" "ts1 shape 1"; do for deg in sr_bicubic deblur_gauss inpainting jpeg deblur_nl; do for sh in 0:25 25:50 50:75 75:100; do jobs_+=("$spec $deg $sh"); done; done; done
lane(){ l=$1; for ((i=l; i<${#jobs_[@]}; i+=3*n)); do set -- ${jobs_[$i]}; v=$1; nz=$2; tilt=$3; deg=$4; sh=$5
  case $deg in sr_bicubic|deblur_gauss|inpainting) tol="ETA=0.01 CGMAX=200";; *) tol="";; esac     # tight solve for linear operators only
  env CUDA_VISIBLE_DEVICES=${G[$((l % n))]} DATA=$D SHARD=$sh SAMPLE_SEED=1 MODE=unified RESTART=never NOISE=$nz TILT=$tilt $tol $P $L/unified.py fin_${v}_$deg --deg=$deg > $L/logs_x/${v}_${deg}_s${sh%%:*}.out 2>&1
done; }
for l in $(seq 0 $((3*n-1))); do lane $l & done; wait; echo ALLDONE
