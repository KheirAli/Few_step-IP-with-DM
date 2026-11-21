#!/bin/bash
# Extrapolation with spatially varying weights, calibrated on 512 MODEL-GENERATED images (no real data), tested on the 107 test images.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/lle
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=prior LINEXACT=1" [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2" )
TASKS=${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl hdr quant onebit prmask}; MODES=${MODES:-scalar pixs pix blk8}
g=0; runone() { local tag=$1 D=$2 SH=$3 task=$4 mode=$5; c=(${CFG[$task]}); deg=${c[0]}; for sh in $SH; do
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} LLE=$(realpath $L/lle)/px_${mode}_${task}.pt SAVE_STEPS=1 $P $L/unified.py fin_llepx${tag}_${mode}_${task}_$deg --deg=$deg > $L/logs_x/llepx${tag}_${mode}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; done; }
rm -f $L/lle/px_*.pt
for K in 1 2 3; do for task in $TASKS; do for mode in $MODES; do runone cal$K $L/data_gen512 "0:256 256:512" $task $mode; done; done; wait
  for task in $TASKS; do c=(${CFG[$task]}); for mode in $MODES; do CUDA_VISIBLE_DEVICES=$((RANDOM%8)) $P $L/lle_fit2.py llepxcal${K}_${mode}_${task}_${c[0]} $L/lle/px_${mode}_${task}.pt $K $mode 2>&1 | grep LLEFIT2; done; done; rm -rf $L/exp_fin_llepxcal${K}_*; done
for task in $TASKS; do for mode in $MODES; do runone test $L/data_paper "0:54 54:107" $task $mode; done; done; wait; echo ALLDONE
