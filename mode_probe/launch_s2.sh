#!/bin/bash
# Two-step version.  SET=sel: all candidate schedules + the repo's own 2-step schedule (native2) + PnP-CM at 2 steps (base2), 24 non-test images.
# SET=test: ONLY_FILE (json task -> schedule name) on the 107 test images, plus PnP-CM at 2 steps.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; SET=${SET:-sel}
if [ $SET = sel ]; then D=$(realpath ../mode_probe/data100); SH="0:24"; else D=$(realpath ../mode_probe/data_paper); SH="0:54 54:107"; fi
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1"
  [jpeg]="jpeg jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg jpeg" [deblur_nl]="deblur_nl deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval inpainting INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl deblur_nl NEWOP=quant" [onebit]="deblur_nl deblur_nl NEWOP=onebit" )
declare -A BASE=( [sr_bicubic]="" [deblur_gauss]="" [inpainting]="" [jpeg]="" [deblur_nl]="" [prmask]="" [hdr]="NEWOP=hdr" [quant]="NEWOP=quant" [onebit]="NEWOP=onebit" )
g=0; go() { g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; }
for task in ${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit}; do set -- ${CFG[$task]}; deg=$1; sch=$2; shift 2
  want=""; [ $SET = test ] && want=$($P -c "import json;print(json.load(open('$L/s2_best.json'))['$task'])")
  for sh in $SH; do
    while read name sig pos; do [ -n "$want" ] && [ "$name" != "$want" ] && continue
      env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 SIGS=$sig SIGPOS=$pos "$@" $P $L/unified.py fin_s2${SET}_${task}_${name}_$deg --deg=$deg > $L/logs_x/s2${SET}_${task}_${name}_${sh%%:*}.out 2>&1 & go
    done < <($P $L/sched2.py $sch)
  done
done; wait; echo ALLDONE
