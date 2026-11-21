#!/bin/bash
# Unmasked phase retrieval, remaining options. 107 images x 3 noise draws. Common: 16 starts, offset, denoiser loop.
#   b16    : batched fits (same method as fin_pn16, timing)                     4 and 8 steps
#   top2   : 4-step sampler from the two best coarse faces, keep the better-fitting final image   (8 calls)
#   same2  : 4-step sampler twice from the best coarse face (different noise), keep the better-fitting (8 calls)
#   top4   : same with four coarse faces (16 calls)
#   cmc... : one model call inside the coarse stage + 3 or 7 sampler steps      (4 and 8 calls)
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe; G=(0 1 2 3 4 5 6 7); NG=${NG:-8}
sg() { $P -c "import numpy as np;print(','.join('%.4f'%v for v in np.geomspace($1,0.03,$2)))"; }
S3=$(sg 0.7 3); S4=$(sg 0.7 4); S7=$(sg 1.5 7); S8=$(sg 1.5 8)
CFG=("b16_4|SIGS=$S4 PRD_BATCH=1" "b16_8|SIGS=$S8 PRD_BATCH=1" "top2_8|SIGS=$S4 PRD_BATCH=1 PRD_TOPK=2" "same2_8|SIGS=$S4 PRD_BATCH=1 PRD_TOPK=2 PRD_TOPSAME=1" "top4_16|SIGS=$S4 PRD_BATCH=1 PRD_TOPK=4"
     "cmcA_4|SIGS=$S3 PRD_BATCH=1 PRD_CMC=0.5 PRD_CMR=32" "cmcA_8|SIGS=$S7 PRD_BATCH=1 PRD_CMC=0.5 PRD_CMR=32" "cmcB_4|SIGS=$S3 PRD_BATCH=1 PRD_CMC=0.3 PRD_CMR=32" "cmcB_8|SIGS=$S7 PRD_BATCH=1 PRD_CMC=0.3 PRD_CMR=32"
     "cmcC_4|SIGS=$S3 PRD_BATCH=1 PRD_CMC=0.5 PRD_CMR=16" "cmcC_8|SIGS=$S7 PRD_BATCH=1 PRD_CMC=0.5 PRD_CMR=16" "cmcD_4|SIGS=$S3 PRD_BATCH=1 PRD_CMC=1.0 PRD_CMR=32" "cmcD_8|SIGS=$S7 PRD_BATCH=1 PRD_CMC=1.0 PRD_CMR=32")
g=0; for ms in ${MSEEDS:-0 1 2}; do for c in "${CFG[@]}"; do name=${c%%|*}; [[ -n "$ONLY" && ! " $ONLY " =~ " ${name%_*} " ]] && continue
  env CUDA_VISIBLE_DEVICES=${G[$((g%NG))]} DATA=$D SHARD=${SHARD:-0:107} SAMPLE_SEED=1 MEAS_SEED=$ms RESTART=never NOISE=white NOLOG=1 NEWOP=prdaps MODE=unified TILT=0 PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_INIT=regfit4 PRD_NSTART=16 PRD_OFFSET=1 PRD_TV=40 ${c#*|} $P $L/unified.py fin_${TAGP:-pa}_${name}_m${ms}_deblur_nl --deg=deblur_nl > $L/logs_x/${TAGP:-pa}_${name}_m${ms}.out 2>&1 &
  g=$((g+1)); if (( g % NG == 0 )); then wait; fi; done; done; wait; echo ALLDONE
