#!/bin/bash
# Linear extrapolation fitted on top of (a) each task's best 4-step configuration and (b) the 2-step base method.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/lle; MODE_=${1:-best}
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1" [jpeg]="jpeg jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg jpeg" [deblur_nl]="deblur_nl deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval inpainting INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1" [hdr]="deblur_nl deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl deblur_nl NEWOP=quant" [onebit]="deblur_nl deblur_nl NEWOP=onebit" )
declare -A BEST=( [inpainting]="FW=1 FWP=0.5" [jpeg]="FW=1 FWP=0.5 FWSTEPS=1,2,3" [deblur_nl]="FW=1 FWP=0.5 FWMIN=0.4" [prmask]="FW=1 FWP=0.5" [hdr]="RHOX=2" [quant]="FW=1 FWP=0.5 FWMIN=0.4" [onebit]="OBMEAN=1.0,0.5,0.3,0.2 FW=1 FWP=0.5 FWMIN=0.4" )
if [ $MODE_ = best ]; then TASKS="inpainting jpeg deblur_nl prmask hdr quant onebit"; KS="1 2 3"; else TASKS="sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit"; KS="1"; fi
extra() { local task=$1; if [ $MODE_ = best ]; then echo "${BEST[$task]}"; else c=(${CFG[$task]}); w=$($P -c "import json;print(json.load(open('$L/s2_best.json'))['$task'])"); a=($($P $L/sched2.py ${c[1]} | grep "^$w ")); echo "SIGS=${a[1]} SIGPOS=${a[2]}"; fi; }
runset() { local tag=$1 D=$2 SH=$3; local g=0; for task in $TASKS; do c=(${CFG[$task]}); deg=${c[0]}; for sh in $SH; do
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:2} $(extra $task) LLE=$(realpath $L/lle)/${MODE_}_${task}.json SAVE_STEPS=1 $P $L/unified.py fin_lle${MODE_}${tag}_${task}_$deg --deg=$deg > $L/logs_x/lle${MODE_}${tag}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % 10 == 0 )); then wait; fi; done; done; wait; }
rm -f $L/lle/${MODE_}_*.json
for K in $KS; do runset sel$K $L/data100 "0:24"
  for task in $TASKS; do c=(${CFG[$task]}); CUDA_VISIBLE_DEVICES=0 $P $L/lle_fit.py lle${MODE_}sel${K}_${task}_${c[0]} $L/lle/${MODE_}_${task}.json $K $([ $task = prmask ] && echo pr) 2>&1 | grep LLEFIT; done; rm -rf $L/exp_fin_lle${MODE_}sel${K}_*; done
runset test $L/data_paper "0:54 54:107"; echo ALLDONE
