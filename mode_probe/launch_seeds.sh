#!/bin/bash
# Clean sampling controls (2026-11-17): same solver, white vs shaped noise, 4 sampler seeds each; plus baseline seeds 3,4.
#   uws{1..4} = unified solver + white noise      shs{3,4} = unified + shaped noise      bs{3,4} = repo baseline
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe
units=()
for v in uws1 uws2 uws3 uws4 shs3 shs4 bs3 bs4; do for sh in 0:25 25:50 50:75 75:100; do units+=("$v $sh"); done; done
lane(){ l=$1; gpu=$((3 + l % 5)); for ((i=l; i<${#units[@]}; i+=10)); do set -- ${units[$i]}; v=$1; sh=$2; seed=${v: -1}
  for deg in sr_bicubic deblur_gauss inpainting jpeg deblur_nl phase_retrieval; do
    case $v in
      uws*) m="MODE=unified RESTART=rule RULE=r2 NOISE=white";;
      shs*) m="MODE=unified RESTART=rule RULE=r2 NOISE=shape";;
      bs*)  m="MODE=base";;
    esac
    env CUDA_VISIBLE_DEVICES=$gpu DATA=$D SHARD=$sh SAMPLE_SEED=$seed $m $P $L/unified.py fin_${v}_$deg --deg=$deg > $L/logs_fin/${v}_${deg}_s${sh%%:*}.out 2>&1
  done; done; }
for l in 0 1 2 3 4 5 6 7 8 9; do lane $l & done; wait; echo ALLDONE
