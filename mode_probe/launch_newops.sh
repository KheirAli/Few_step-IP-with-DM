#!/bin/bash
# 2026-11-17: nonlinear tasks beyond the repo's three.  Settings borrowed from nonlinear deblur for baseline and ours; tilt +0.5.
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; G=(${GPUS:-0 1 2 3 7}); n=${#G[@]}
jobs_=()
add(){ for sh in 0:25 25:50 50:75 75:100; do jobs_+=("$sh $*"); done; }
add quantx NEWOP=quant MODE=unified TILT=0.5;      add quantb NEWOP=quant MODE=base
add onebitx NEWOP=onebit MODE=unified TILT=0.5;    add onebitb NEWOP=onebit MODE=base
add blindx NEWOP=blind MODE=unified TILT=0.5 BLINDK=est;  add blindt NEWOP=blind MODE=unified TILT=0.5 BLINDK=true;  add blindn NEWOP=blind MODE=unified TILT=0.5 BLINDK=none
add hazex NEWOP=haze MODE=unified TILT=0.5 HAZEK=est;     add hazet NEWOP=haze MODE=unified TILT=0.5 HAZEK=true;     add hazen NEWOP=haze MODE=unified TILT=0.5 HAZEK=none
lane(){ l=$1; for ((i=l; i<${#jobs_[@]}; i+=n)); do set -- ${jobs_[$i]}; sh=$1; v=$2; shift 2
  env CUDA_VISIBLE_DEVICES=${G[$l]} DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 "$@" $P $L/unified.py fin_${v}_deblur_nl --deg=deblur_nl > $L/logs_x/${v}_s${sh%%:*}.out 2>&1
done; }
for l in $(seq 0 $((n-1))); do lane $l & done; wait; echo ALLDONE
