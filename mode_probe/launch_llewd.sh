#!/bin/bash
# Structured combinations fitted ON TOP OF EACH TASK'S BEST CONFIGURATION (frequency-weighted data-fit where used; HDR with first-step sigma 3.0).
# Calibration on 512 MODEL-GENERATED images (no real data), test on the 107 test images, 4 calls.
# modes: scalar | blk8 (8x8 blocks) | pixs (per pixel) | pix (per pixel and colour) | dct30 (30 smooth maps) | eig30 (30-dim face embedding + one weight for the rest)
# env: NGPU (8), PAR (2*NGPU), TASKS, MODES, CALSH ("0:256 256:512"), TESTSH ("0:54 54:107"), TAG (llew)
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/lle $L/logs_x; NGPU=${NGPU:-8}; PAR=${PAR:-$((2*NGPU))}; TAG=${TAG:-llew}
declare -A CFG=( [$KEY]="$CFGSTR" ); NSTEP=${NSTEP:-4}
TASKS=$KEY; MODES=${MODES:-scalar blk8 pixs pix dct30 eig30}
g=${GOFF:-0}; runone() { local tag=$1 D=$2 SH=$3 task=$4 mode=$5; c=(${CFG[$task]}); deg=${c[0]}; for sh in $SH; do
  env CUDA_VISIBLE_DEVICES=$((g%NGPU)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} LLE=$(realpath $L/lle)/${TAG}_${mode}_${task}.pt SAVE_STEPS=1 $P $L/unified.py fin_${TAG}${tag}_${mode}_${task}_$deg --deg=$deg > $L/logs_x/${TAG}${tag}_${mode}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % PAR == 0 )); then wait; fi; done; }
fit() { local K=$1 src=$2 task=$3 mode=$4; c=(${CFG[$task]}); CUDA_VISIBLE_DEVICES=$(( ${GOFF:-0} % NGPU )) $P $L/lle_fit2.py ${TAG}cal${K}_${src}_${task}_${c[0]} $L/lle/${TAG}_${mode}_${task}.pt $K $mode $([ $task = prmask ] && echo pr) 2>&1 | grep -E "LLEFIT2|Error"; }
rm -f $L/lle/${TAG}_*.pt
# round 1: no weights exist yet, so one run ("base") serves every mode
for task in $TASKS; do runone cal1 ${CALDATA:-$L/data_gen512} "${CALSH:-0:256 256:512}" $task base; done; wait
for task in $TASKS; do for mode in $MODES; do fit 1 base $task $mode; done; done; rm -rf $L/exp_fin_${TAG}cal1_*
for K in $(seq 2 $((NSTEP-1))); do for task in $TASKS; do for mode in $MODES; do runone cal$K ${CALDATA:-$L/data_gen512} "${CALSH:-0:256 256:512}" $task $mode; done; done; wait
  for task in $TASKS; do for mode in $MODES; do fit $K $mode $task $mode; done; done; rm -rf $L/exp_fin_${TAG}cal${K}_*; done
for task in $TASKS; do for mode in $MODES; do runone test ${TESTDATA:-$L/data_paper} "${TESTSH:-0:54 54:107}" $task $mode; done; done; wait; echo ALLDONE
