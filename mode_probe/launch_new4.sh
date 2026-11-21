#!/bin/bash
# Four more separable nonlinear operators on the 107 test images.  Settings borrowed from nonlinear deblur; NOTHING tuned per task.
#   ours4 : closed-form data-fit, 4 calls      ours4w : + frequency weighting at steps 2-4 (the single rule)      ours2 : 2 calls (first and last shipped levels)
#   pnp4  : PnP-CM's own gradient solver, 4 calls
cd /usr/scratch/akheirandish3/PNP/PnP-CM; D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe; export OMP_NUM_THREADS=4; g=0
for op in ${OPS:-gamma mod blurclip blurq}; do for v in "ours4|MODE=unified TILT=0.5" "ours4w|MODE=unified TILT=0.5 FW=1 FWP=0.5 FWSTEPS=1,2,3" "ours2|MODE=unified TILT=0.5 SIGS=0.6940,0.4590 SIGPOS=0,3" "pnp4|MODE=unified TILT=0 SOLVER=repo"; do
  for sh in ${SHS:-0:54 54:107}; do env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 NEWOP=$op ${v#*|} $P $L/unified.py fin_${TAGP:-n4}_${v%%|*}_${op} --deg=deblur_nl > $L/logs_x/${TAGP:-n4}_${v%%|*}_${op}_${sh%%:*}.out 2>&1 &
    g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; done; done; done; wait; echo ALLDONE
