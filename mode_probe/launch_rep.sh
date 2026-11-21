#!/bin/bash
# Diversity term test: particle 2 (seed 2) is pushed away from particle 1's finished image inside every data-fit. 107 test images, 4 steps.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit" )
g=0
for task in sr_bicubic deblur_gauss inpainting jpegtrue deblur_nl hdr quant onebit; do set -- ${CFG[$task]}; deg=$1; shift
  for gg in ${GS:-0.3 0.6}; do for sh in 0:54 54:107; do
    env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=2 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 REPEL_FROM=fin_s8test_${task}_ship4_${deg} REPEL_G=$gg "$@" $P $L/unified.py fin_rep${gg}_${task}_$deg --deg=$deg > $L/logs_x/rep${gg}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1))
    if (( g % 16 == 0 )); then wait; fi; done; done
done; wait; echo ALLDONE
