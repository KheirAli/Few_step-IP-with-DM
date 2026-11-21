#!/bin/bash
# Sparse-view CT (parallel beam, 60 and 30 views) on face images with the face consistency model: an OPERATOR test of the method.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; export OMP_NUM_THREADS=4; g=0
C="SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified NEWOP=ct"
run() { local tag=$1 set=$2; shift 2; local D SH; case $set in sel) D=$L/data100; SH="0:24";; test) D=$L/data_paper; SH="0:54 54:107";; esac
  for sh in $SH; do env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh $C "$@" $P $L/unified.py fin_ct${set}_${tag} --deg=deblur_nl > $L/logs_x/ct${set}_${tag}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; done; }
if [ "${STAGE:-sel}" = sel ]; then
  for rx in 0.1 0.3 1 3; do run nl_rx$rx sel TILT=0.5 RHOX=$rx; run lin_rx$rx sel TILT=0.5 RHOX=$rx SIGS=0.327,0.132,0.02,0.01; run mid_rx$rx sel TILT=0.5 RHOX=$rx SIGS=0.5,0.25,0.1,0.02; done; wait
else
  for K in 60 30; do B="CT_K=$K $BEST"
    run k${K}_white test TILT=0 $B; run k${K}_tilt test TILT=0.5 $B; run k${K}_fw test TILT=0.5 FW=1 FWP=0.5 FWSTEPS=1,2,3 $B; run k${K}_2step test TILT=0.5 CT_K=$K $B2; done; wait
fi; echo ALLDONE
