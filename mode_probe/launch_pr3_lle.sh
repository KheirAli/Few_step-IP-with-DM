#!/bin/bash
# Linear extrapolation for unmasked phase retrieval: fitted step by step on the 24 selection images, for 4 and 8 steps.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/lle
sg() { $P -c "import numpy as np;print(','.join('%.4f'%v for v in np.geomspace($1,0.03,$2)))"; }; S4=$(sg 0.7 4); S8=$(sg 1.5 8)
C="SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 NEWOP=prdaps MODE=unified PRD_K=0 PRD_FINAL=0 TILT=0 PRD_SOFT=1 PRD_INIT=regfit4 PRD_NSTART=16 PRD_BATCH=1 PRD_OFFSET=1 PRD_TV=40 SAVE_STEPS=1 $EXTRA"
rm -f $L/lle/pru${SUF}_4.json $L/lle/pru${SUF}_8.json
for K in 1 2 3 4 5 6 7; do
  [ $K -le 3 ] && env CUDA_VISIBLE_DEVICES=0 DATA=$(realpath $L/data100) SHARD=0:24 MEAS_SEED=0 $C SIGS=$S4 LLE=$(realpath $L/lle)/pru${SUF}_4.json $P $L/unified.py fin_prl${SUF}sel4_deblur_nl --deg=deblur_nl > $L/logs_x/prl${SUF}sel4.out 2>&1 &
  env CUDA_VISIBLE_DEVICES=1 DATA=$(realpath $L/data100) SHARD=0:24 MEAS_SEED=0 $C SIGS=$S8 LLE=$(realpath $L/lle)/pru${SUF}_8.json $P $L/unified.py fin_prl${SUF}sel8_deblur_nl --deg=deblur_nl > $L/logs_x/prl${SUF}sel8.out 2>&1 & wait
  [ $K -le 3 ] && CUDA_VISIBLE_DEVICES=0 $P $L/lle_fit.py prl${SUF}sel4_deblur_nl $L/lle/pru${SUF}_4.json $K 2>&1 | grep LLEFIT
  CUDA_VISIBLE_DEVICES=0 $P $L/lle_fit.py prl${SUF}sel8_deblur_nl $L/lle/pru${SUF}_8.json $K 2>&1 | grep LLEFIT; rm -rf $L/exp_fin_prl${SUF}sel*; done
g=2; for ms in 0 1 2; do for st in 4 8; do [ $st = 4 ] && SG=$S4 || SG=$S8
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $L/data_paper) SHARD=0:107 MEAS_SEED=$ms $C SIGS=$SG LLE=$(realpath $L/lle)/pru${SUF}_$st.json $P $L/unified.py fin_prl${SUF}test${st}_m${ms}_deblur_nl --deg=deblur_nl > $L/logs_x/prl${SUF}test${st}_m$ms.out 2>&1 & g=$((g+1)); done; done; wait; echo ALLDONE
