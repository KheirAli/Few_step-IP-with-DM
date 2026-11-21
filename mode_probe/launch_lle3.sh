#!/bin/bash
# Does the extrapolation need learning?  (a) fixed rules, no fitting;  (b) coefficients fitted on 24 MODEL-GENERATED images (no real data);
# (c) one set of coefficients shared by all tasks of a schedule family (fitted on HDR, applied to the others).  4 steps, base method, 107 test images.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1" [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit" )
TASKS="sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit"
runset() { local tag=$1 D=$2 SH=$3 coef=$4; local g=0; for task in $TASKS; do c=(${CFG[$task]}); deg=${c[0]}; cf=${coef//TASK/$task}; for sh in $SH; do
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} LLE=$(realpath $L/lle)/$cf SAVE_STEPS=1 $P $L/unified.py fin_lle${tag}_${task}_$deg --deg=$deg > $L/logs_x/lle${tag}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % 10 == 0 )); then wait; fi; done; done; wait; }
rm -f $L/lle/gen_*.json
for K in 1 2 3; do runset gensel$K $L/data_gen "0:24" gen_TASK.json
  for task in $TASKS; do c=(${CFG[$task]}); CUDA_VISIBLE_DEVICES=0 $P $L/lle_fit.py llegensel${K}_${task}_${c[0]} $L/lle/gen_${task}.json $K $([ $task = prmask ] && echo pr) > /dev/null 2>&1; done; rm -rf $L/exp_fin_llegensel${K}_*; done
runset gentest $L/data_paper "0:54 54:107" gen_TASK.json
runset fixavg $L/data_paper "0:54 54:107" fixed_avg.json; runset fixlast $L/data_paper "0:54 54:107" fixed_last.json
cp $L/lle/hdr.json $L/lle/shared_hdr.json; runset shared $L/data_paper "0:54 54:107" shared_hdr.json; echo ALLDONE
