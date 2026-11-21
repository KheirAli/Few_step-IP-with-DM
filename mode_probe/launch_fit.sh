#!/bin/bash
# Through-model measurement fit as a full method (2026-11-17): 100 images x 4 seeds.
#   fit1 / fit3 = phase retrieval, fit at the last 1 / 3 steps of the second pass (step 0.1 x level, 5 iterations)
#   jfit1       = JPEG, last step, small steps (0.02 x level)
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; G=(${GPUS:-0 3 4 7}); n=${#G[@]}
jobs_=()
for seed in 1 2 3 4; do for sh in 0:25 25:50 50:75 75:100; do
  jobs_+=("fit1 phase_retrieval $seed $sh FITK=1 FITPASS=1 TETA=0.1 RESTART=rule RULE=r2")
  jobs_+=("fit3 phase_retrieval $seed $sh FITK=3 FITPASS=1 TETA=0.1 RESTART=rule RULE=r2")
  jobs_+=("jfit1 jpeg $seed $sh FITK=1 FITPASS=0 TETA=0.02 RESTART=rule RULE=r2")
done; done
lane(){ l=$1; for ((i=l; i<${#jobs_[@]}; i+=3*n)); do set -- ${jobs_[$i]}; v=$1; deg=$2; seed=$3; sh=$4; shift 4
  env CUDA_VISIBLE_DEVICES=${G[$((l % n))]} DATA=$D SHARD=$sh SAMPLE_SEED=$seed MODE=unified NOISE=white "$@" $P $L/unified.py fin_${v}${seed}_$deg --deg=$deg > $L/logs_x/${v}${seed}_${deg}_s${sh%%:*}.out 2>&1
done; }
for l in $(seq 0 $((3*n-1))); do lane $l & done; wait; echo ALLDONE
