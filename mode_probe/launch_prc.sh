#!/bin/bash
# Masked phase retrieval: 4 classical solutions per image, each completed by the 2-step sampler; selection without the truth. 107 test images.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe; export OMP_NUM_THREADS=4; g=0
a=($($P $L/sched2.py inpainting | grep "^p05_2 "))
for v in "k4|HIO_K=4" "k1|HIO_K=1"; do for sh in 0:27 27:54 54:81 81:107; do
  env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 INIT=hio HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1 PRCAND=4 ${v#*|} SIGS=${a[1]} SIGPOS=${a[2]} $P $L/unified.py fin_prc_${v%%|*} --deg=phase_retrieval > $L/logs_x/prc_${v%%|*}_${sh%%:*}.out 2>&1 & g=$((g+1)); done; done; wait; echo ALLDONE
