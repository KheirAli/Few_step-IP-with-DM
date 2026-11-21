#!/bin/bash
# MRI-style undersampled k-space (single coil, Cartesian) on face images with the face model: an OPERATOR test.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; export OMP_NUM_THREADS=4; g=0
C="SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified NEWOP=mri"
run() { local tag=$1 set=$2; shift 2; local D SH; case $set in sel) D=$L/data100; SH="0:24";; test) D=$L/data_paper; SH="0:54 54:107";; esac
  for sh in $SH; do env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh $C "$@" $P $L/unified.py fin_mri${set}_${tag} --deg=deblur_nl > $L/logs_x/mri${set}_${tag}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; done; }
if [ "${STAGE:-sel}" = sel ]; then
  for rx in 0.1 0.3 1 3 10; do run nl_rx$rx sel TILT=0.5 RHOX=$rx; run lin_rx$rx sel TILT=0.5 RHOX=$rx SIGS=0.327,0.132,0.02,0.01; run mid_rx$rx sel TILT=0.5 RHOX=$rx SIGS=0.5,0.25,0.1,0.02; done
  for rx in 0.3 1 3; do for s2 in 0.3,0.05 0.5,0.1; do run two${s2/,/-}_rx$rx sel TILT=0.5 RHOX=$rx SIGS=$s2; done; done; wait
else
  for R in 4 8; do run r${R}_white test MRI_R=$R TILT=0 $BEST; run r${R}_tilt test MRI_R=$R TILT=0.5 $BEST; run r${R}_fw test MRI_R=$R TILT=0.5 FW=1 FWP=0.5 FWSTEPS=1,2,3 $BEST; run r${R}_2step test MRI_R=$R TILT=0.5 $B2; done; wait
fi; echo ALLDONE
