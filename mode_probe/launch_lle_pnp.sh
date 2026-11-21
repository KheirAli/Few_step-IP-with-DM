#!/bin/bash
# PnP-CM (its own data-fit solver, white noise, its own start, shipped schedule) WITH linear extrapolation, fitted exactly as for ours:
# step by step on the 24 selection images, then applied once to the 107 test images.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/lle
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpeg]="jpeg SOLVER=repo" [deblur_nl]="deblur_nl SOLVER=repo"
  [hdr]="deblur_nl NEWOP=hdr SOLVER=repo" [quant]="deblur_nl NEWOP=quant SOLVER=repo" [onebit]="deblur_nl NEWOP=onebit SOLVER=repo" [prmask]="phase_retrieval SOLVER=repo" )
TASKS=${TASKS:-sr_bicubic deblur_gauss inpainting jpeg deblur_nl hdr quant onebit prmask}
runset() { local tag=$1 D=$2 SH=$3 only=$4; local g=0; for task in $TASKS; do [ -n "$only" ] && [ "$only" != "$task" ] && continue; c=(${CFG[$task]}); deg=${c[0]}; for sh in $SH; do
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0 ${c[@]:1} LLE=$(realpath $L/lle)/pnp_${task}.json SAVE_STEPS=1 $P $L/unified.py fin_llepnp${tag}_${task}_$deg --deg=$deg > $L/logs_x/llepnp${tag}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % 12 == 0 )); then wait; fi; done; done; wait; }
rm -f $L/lle/pnp_*.json
for K in 1 2 3 4 5 6 7; do only=""; [ $K -gt 3 ] && only=prmask
  runset sel$K $L/data100 "0:24" $only
  for task in $TASKS; do [ -n "$only" ] && [ "$only" != "$task" ] && continue; c=(${CFG[$task]}); CUDA_VISIBLE_DEVICES=0 $P $L/lle_fit.py llepnpsel${K}_${task}_${c[0]} $L/lle/pnp_${task}.json $K $([ $task = prmask ] && echo pr) 2>&1 | grep LLEFIT; done; rm -rf $L/exp_fin_llepnpsel${K}_*; done
runset test $L/data_paper "0:54 54:107"; echo ALLDONE
