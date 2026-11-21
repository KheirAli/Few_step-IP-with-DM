#!/bin/bash
# Unmasked phase retrieval with learned low-frequency phases as soft constraints (network trained on model samples only). 107 images x 3 noise draws.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe; G=(0 1 2 3 4 5 6 7)
S4=$($P -c "import numpy as np;print(','.join('%.4f'%v for v in np.geomspace(0.7,0.03,4)))"); S8=$($P -c "import numpy as np;print(','.join('%.4f'%v for v in np.geomspace(1.5,0.03,8)))")
g=0; for cfg in "net8 0" "net8r8 8"; do set -- $cfg; for ms in 0 1 2; do for st in 4 8; do [ $st = 4 ] && SG=$S4 || SG=$S8
  env CUDA_VISIBLE_DEVICES=${G[$((g%8))]} DATA=$D SHARD=0:107 SAMPLE_SEED=1 MEAS_SEED=$ms RESTART=never NOISE=white NOLOG=1 NEWOP=prdaps MODE=unified TILT=0 PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_INIT=netfit PRD_NET=k8biglong PRD_NETREG=$2 PRD_OFFSET=1 PRD_TV=40 SIGS=$SG $P $L/unified.py fin_p${1}_${st}_m${ms}_deblur_nl --deg=deblur_nl > $L/logs_x/p${1}_${st}_m${ms}.out 2>&1 &
  g=$((g+1)); if (( g % 6 == 0 )); then wait; fi; done; done; done; wait; echo ALLDONE
