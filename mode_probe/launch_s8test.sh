#!/bin/bash
# The 8-step schedule selected per task on non-test images (s8_best.json), run once on the paper's 107 test images.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe; G=(0 1 2 3 4 5 6 7)
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1"
  [jpeg]="jpeg jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg jpeg" [deblur_nl]="deblur_nl deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval inpainting INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl deblur_nl NEWOP=quant" [onebit]="deblur_nl deblur_nl NEWOP=onebit" )
g=0
for task in sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit; do set -- ${CFG[$task]}; deg=$1; sch=$2; shift 2
  best=$($P -c "import json;print(json.load(open('$L/s8_best.json'))['$task'])")
  for name in ship4 $best; do line=$($P $L/sched8.py $sch | grep "^$name "); set2=($line); extra="SIGS=${set2[1]} SIGPOS=${set2[2]}"; [ "$name" = ship4 ] && extra=""
    for sh in 0:54 54:107; do env CUDA_VISIBLE_DEVICES=${G[$((g%8))]} DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 $extra "$@" $P $L/unified.py fin_s8test_${task}_${name}_$deg --deg=$deg > $L/logs_x/s8test_${task}_${name}_${sh%%:*}.out 2>&1 & g=$((g+1)); done
    if (( g % 16 == 0 )); then wait; fi
  done
done; wait; echo ALLDONE
