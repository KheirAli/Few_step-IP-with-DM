#!/bin/bash
# 2026-11-17: blur model-error correction and soft JPEG interval projection, 100 images, seed 1, 4 model calls.
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; G=(${GPUS:-0 1 2 3}); n=${#G[@]}
jobs_=()
add(){ for sh in 0:25 25:50 50:75 75:100; do jobs_+=("$sh $*"); done; }
add mec1w deblur_nl WARMX=lin NLKER=1 MEC=once;            add mec1t deblur_nl WARMX=lin NLKER=1 MEC=once TILT=0.5
add mecsw deblur_nl WARMX=lin NLKER=1 MEC=step;            add mecst deblur_nl WARMX=lin NLKER=1 MEC=step TILT=0.5
add ckbase deblur_nl MODE=base;                            add ckwarm deblur_nl WARMX=1;      add cklin deblur_nl WARMX=lin NLKER=1
add jsw jpeg JPROJ=1 JSOFT=1;                              add jst jpeg JPROJ=1 JSOFT=1 TILT=0.5
add jscw jpeg JPROJ=1 JSOFT=1 JCOLD=1;                     add jsct jpeg JPROJ=1 JSOFT=1 JCOLD=1 TILT=0.5
add ckbase jpeg MODE=base;                                 add cktrue jpeg TILT=0.5;          add ckhard jpeg JPROJ=1 TILT=0.5
lane(){ l=$1; for ((i=l; i<${#jobs_[@]}; i+=4*n)); do set -- ${jobs_[$i]}; sh=$1; v=$2; deg=$3; shift 3
  env CUDA_VISIBLE_DEVICES=${G[$((l % n))]} DATA=$D SHARD=$sh SAMPLE_SEED=1 MODE=unified RESTART=never NOISE=white "$@" $P $L/unified.py fin_${v}_$deg --deg=$deg > $L/logs_x/${v}_${deg}_s${sh%%:*}.out 2>&1
done; }
for l in $(seq 0 $((4*n-1))); do lane $l & done; wait; echo ALLDONE
