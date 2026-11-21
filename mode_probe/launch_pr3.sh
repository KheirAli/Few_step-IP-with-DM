#!/bin/bash
# Unmasked phase retrieval, 24 selection images (data100), one noise draw.
#   A. PnP-CM's own gradient solver on this task (it ships no setting for it): starts x iterations x learning rates
#   B. ours + expected-value magnitude fit     C. ours + frequency-weighting rule     (extrapolation is fitted by launch_pr3_lle.sh)
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data100); P=../pnpcm_env/bin/python; L=../mode_probe; g=0
sg() { $P -c "import numpy as np;print(','.join('%.4f'%v for v in np.geomspace($1,0.03,$2)))"; }; S4=$(sg 0.7 4); S8=$(sg 1.5 8)
C="DATA=$D SHARD=0:24 SAMPLE_SEED=1 MEAS_SEED=0 RESTART=never NOISE=white NOLOG=1 NEWOP=prdaps MODE=unified PRD_K=0 PRD_FINAL=0"
OURS="TILT=0 PRD_SOFT=1 PRD_INIT=regfit4 PRD_NSTART=16 PRD_BATCH=1 PRD_OFFSET=1 PRD_TV=40"
run() { local tag=$1; shift; env CUDA_VISIBLE_DEVICES=$((g%8)) $C "$@" $P $L/unified.py fin_pr3_${tag}_deblur_nl --deg=deblur_nl > $L/logs_x/pr3_$tag.out 2>&1 & g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; }
if [[ " ${PARTS:-A B C} " =~ " A " ]]; then
  for st in "cold|" "hio|PRD_K=1 PRD_HIO=500" "ours|PRD_INIT=regfit4 PRD_NSTART=16 PRD_BATCH=1 PRD_OFFSET=1 PRD_TV=40"; do for n in 100 1000; do for lr in 5e-4 5e-3 5e-2; do
    run pnp_${st%%|*}_n${n}_lr$lr TILT=0 SOLVER=repo REPO_N=$n REPO_LR=$lr SIGS=$S8 ${st#*|}; done; done; done; wait; fi
if [[ " ${PARTS:-A B C} " =~ " B " ]]; then
  run base4 $OURS SIGS=$S4; run base8 $OURS SIGS=$S8
  for ev in 0.5 1.0 2.0 "2.0,1.0,0.5,0.5" ; do run ev${ev//,/-}_4 $OURS SIGS=$S4 PRD_EV=$ev; done
  for ev in 0.5 1.0 2.0 "2.0,2.0,1.0,1.0,0.5,0.5,0.5,0.5"; do run ev${ev//,/-}_8 $OURS SIGS=$S8 PRD_EV=$ev; done; fi
if [[ " ${PARTS:-A B C} " =~ " C " ]]; then
  run fw_4 $OURS SIGS=$S4 FW=1 FWP=0.5 FWSTEPS=1,2,3; run fw_8 $OURS SIGS=$S8 FW=1 FWP=0.5 FWSTEPS=1,2,3,4,5,6,7; run tilt_4 $OURS TILT=0.5 SIGS=$S4; run tilt_8 $OURS TILT=0.5 SIGS=$S8; fi
wait; echo ALLDONE
