#!/bin/bash
# Unmasked phase retrieval, full pipeline on the paper's 107 images with 3 measurement-noise draws: 1 start vs 4 starts, 4 and 8 steps.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe; G=(0 1 2 3 4 5 6 7)
S4=$($P -c "import numpy as np;print(','.join('%.4f'%v for v in np.geomspace(0.7,0.03,4)))"); S8=$($P -c "import numpy as np;print(','.join('%.4f'%v for v in np.geomspace(1.5,0.03,8)))")
g=0; for ms in 0 1 2; do for init in regfit regfit4; do for st in 4 8; do [ $st = 4 ] && SG=$S4 || SG=$S8
  env CUDA_VISIBLE_DEVICES=${G[$((g%8))]} DATA=$D SHARD=0:107 SAMPLE_SEED=1 MEAS_SEED=$ms RESTART=never NOISE=white NOLOG=1 NEWOP=prdaps MODE=unified TILT=0 PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_INIT=$init SIGS=$SG $P $L/unified.py fin_pq_${init}_${st}_m${ms}_deblur_nl --deg=deblur_nl > $L/logs_x/pq_${init}_${st}_m${ms}.out 2>&1 &
  g=$((g+1)); done; done; done; wait; echo ALLDONE
