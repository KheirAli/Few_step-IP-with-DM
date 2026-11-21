#!/bin/bash
# 2026-11-17: every data-fit in closed form, the SAME noise tilt (+0.5) on every operator, baseline's model-call budget or less.
# One job per GPU so the timings are not distorted by our own jobs sharing a card.  Baseline rerun the same way for timing.
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; G=(${GPUS:-0 1 2 3 7}); n=${#G[@]}
jobs_=()
add(){ for sh in 0:25 25:50 50:75 75:100; do jobs_+=("$sh $*"); done; }
for deg in sr_bicubic deblur_gauss inpainting; do add ex05 $deg MODE=unified LINEXACT=1 TILT=0.5; done
add ex05 deblur_nl MODE=unified WARMX=lin NLKER=1 MEC=step TILT=0.5
add ex05 jpeg MODE=unified JPROJ=1 JSOFT=1 JCOLD=1 TILT=0.5
add ex05 phase_retrieval MODE=unified INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1 TILT=0.5
for deg in sr_bicubic deblur_gauss inpainting deblur_nl jpeg phase_retrieval; do add tbase $deg MODE=base; done
lane(){ l=$1; for ((i=l; i<${#jobs_[@]}; i+=n)); do set -- ${jobs_[$i]}; sh=$1; v=$2; deg=$3; shift 3
  env CUDA_VISIBLE_DEVICES=${G[$l]} DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 "$@" $P $L/unified.py fin_${v}_$deg --deg=$deg > $L/logs_x/${v}_${deg}_s${sh%%:*}.out 2>&1
done; }
for l in $(seq 0 $((n-1))); do lane $l & done; wait; echo ALLDONE
