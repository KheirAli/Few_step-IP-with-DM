#!/bin/bash
# Learnable linear extrapolation on top of our 4-step method.  Coefficients are fitted step by step on the 24 selection images
# (three short rounds), then applied once to the 107 test images.  No extra model calls.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/lle
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1" [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit" )
TASKS=${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit}
runset() { local tag=$1 D=$2 SH=$3; local g=0; for task in $TASKS; do c=(${CFG[$task]}); deg=${c[0]}; for sh in $SH; do
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} LLE=$(realpath $L/lle)/${task}.json SAVE_STEPS=1 $P $L/unified.py fin_lle${tag}_${task}_$deg --deg=$deg > $L/logs_x/lle${tag}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % 10 == 0 )); then wait; fi; done; done; wait; }
rm -f $L/lle/*.json
for K in 1 2 3; do runset sel$K $L/data100 "0:24"
  for task in $TASKS; do c=(${CFG[$task]}); CUDA_VISIBLE_DEVICES=0 $P $L/lle_fit.py llesel${K}_${task}_${c[0]} $L/lle/${task}.json $K $([ $task = prmask ] && echo pr) 2>&1 | grep LLEFIT; done; rm -rf $L/exp_fin_llesel${K}_*; done
runset sel $L/data100 "0:24"; runset test $L/data_paper "0:54 54:107"; echo ALLDONE
