#!/bin/bash
# SERVER-1 JOB 2: extrapolation weights that vary over the image or over a 30-dimensional embedding, against plain scalar weights.
# Calibration on 512 MODEL-GENERATED images (no real data), test on the 107 test images, 4 calls.
# modes: scalar | blk8 (8x8 blocks) | pixs (per pixel) | pix (per pixel and colour) | dct30 (30 smooth maps) | eig30 (30-dim face embedding + one weight for the rest)
# env: NGPU (8), PAR (2*NGPU), TASKS, MODES, CALSH ("0:256 256:512"), TESTSH ("0:54 54:107"), TAG (llew)
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/lle $L/logs_x; NGPU=${NGPU:-8}; PAR=${PAR:-$((2*NGPU))}; TAG=${TAG:-llew}
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=prior LINEXACT=1" [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2" )
TASKS=${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl hdr quant onebit prmask}; MODES=${MODES:-scalar blk8 pixs pix dct30 eig30}
GL=(0 1 2 4 5 7); g=0; runone() { local tag=$1 D=$2 SH=$3 task=$4 mode=$5; c=(${CFG[$task]}); deg=${c[0]}; for sh in $SH; do
  env CUDA_VISIBLE_DEVICES=${GL[$((g%6))]} DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} LLE=$(realpath $L/lle)/${TAG}_${mode}_${task}.pt SAVE_STEPS=1 $P $L/unified.py fin_${TAG}${tag}_${mode}_${task}_$deg --deg=$deg > $L/logs_x/${TAG}${tag}_${mode}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % PAR == 0 )); then wait; fi; done; }
fit() { local K=$1 src=$2 task=$3 mode=$4; c=(${CFG[$task]}); CUDA_VISIBLE_DEVICES=4 $P $L/lle_fit2.py ${TAG}cal${K}_${src}_${task}_${c[0]} $L/lle/${TAG}_${mode}_${task}.pt $K $mode 2>&1 | grep -E "LLEFIT2|Error"; }
rm -f $L/lle/${TAG}_*.pt
# round 1: no weights exist yet, so one run ("base") serves every mode
for task in $TASKS; do runone cal1 $L/data_gen512 "${CALSH:-0:256 256:512}" $task base; done; wait
for task in $TASKS; do for mode in $MODES; do fit 1 base $task $mode; done; done; rm -rf $L/exp_fin_${TAG}cal1_*
for K in 2 3; do for task in $TASKS; do for mode in $MODES; do runone cal$K $L/data_gen512 "${CALSH:-0:256 256:512}" $task $mode; done; done; wait
  for task in $TASKS; do for mode in $MODES; do fit $K $mode $task $mode; done; done; rm -rf $L/exp_fin_${TAG}cal${K}_*; done
for task in $TASKS; do for mode in $MODES; do runone test $L/data_paper "${TESTSH:-0:54 54:107}" $task $mode; done; done; wait; echo ALLDONE
