#!/bin/bash
# 2026-11-17: HDR clipping (DPS/DAPS definition: clip(2x)), a nonlinear task the repo does not ship.  Settings borrowed from nonlinear
# deblur for BOTH baseline and ours (nothing tuned).  One job per GPU for timing.
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; G=(${GPUS:-0 1 2 3 7}); n=${#G[@]}
jobs_=()
add(){ for sh in 0:25 25:50 50:75 75:100; do jobs_+=("$sh $*"); done; }
add hdrx MODE=unified HDRX=1;          add hdrxt MODE=unified HDRX=1 TILT=0.5
add hdrgn MODE=unified;                add hdrgnt MODE=unified TILT=0.5
add hdrbase MODE=base
lane(){ l=$1; for ((i=l; i<${#jobs_[@]}; i+=n)); do set -- ${jobs_[$i]}; sh=$1; v=$2; shift 2
  env CUDA_VISIBLE_DEVICES=${G[$l]} DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 NEWOP=hdr "$@" $P $L/unified.py fin_${v}_deblur_nl --deg=deblur_nl > $L/logs_x/${v}_s${sh%%:*}.out 2>&1
done; }
for l in $(seq 0 $((n-1))); do lane $l & done; wait; echo ALLDONE
