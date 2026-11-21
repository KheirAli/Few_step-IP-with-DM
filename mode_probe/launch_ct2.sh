#!/bin/bash
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; export OMP_NUM_THREADS=4; g=0
C="SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified NEWOP=ct DATA=$(realpath $L/data100) SHARD=0:24 TILT=0.5"
for rx in 10 30 100; do env CUDA_VISIBLE_DEVICES=$((g%8)) $C RHOX=$rx SIGS=0.5,0.25,0.1,0.02 $P $L/unified.py fin_ctsel_mid_rx$rx --deg=deblur_nl > $L/logs_x/ctsel_mid_rx$rx.out 2>&1 & g=$((g+1))
  env CUDA_VISIBLE_DEVICES=$((g%8)) $C RHOX=$rx SIGS=0.327,0.132,0.02,0.01 $P $L/unified.py fin_ctsel_lin_rx$rx --deg=deblur_nl > $L/logs_x/ctsel_lin_rx$rx.out 2>&1 & g=$((g+1)); done
for s2 in 0.5,0.1 0.5,0.02 0.3,0.05; do for rx in 3 10; do env CUDA_VISIBLE_DEVICES=$((g%8)) $C RHOX=$rx SIGS=$s2 $P $L/unified.py fin_ctsel_two${s2/,/-}_rx$rx --deg=deblur_nl > $L/logs_x/ctsel_two_$g.out 2>&1 & g=$((g+1)); done; done; wait
