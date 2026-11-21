#!/bin/bash
# Choose an 8-step noise schedule per task on 24 images that are NOT in the test set (data100), then nothing else: selection only.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/${DSET:-data100}); P=../pnpcm_env/bin/python; L=../mode_probe; G=(0 1 2 3 4 5 6 7); N=${N:-24}; TAG=${TAG:-s8sel}
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1"
  [jpeg]="jpeg jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg jpeg" [deblur_nl]="deblur_nl deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval inpainting INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl deblur_nl NEWOP=quant" [onebit]="deblur_nl deblur_nl NEWOP=onebit" )
g=0
for task in ${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit}; do set -- ${CFG[$task]}; deg=$1; sch=$2; shift 2
  while read name sig pos; do [ -n "$ONLY" ] && [ "$name" != "${ONLY_MAP[$task]:-$ONLY}" ] && continue
    extra="SIGS=$sig SIGPOS=$pos"; [ "$name" = ship4 ] && extra=""
    env CUDA_VISIBLE_DEVICES=${G[$((g%8))]} DATA=$D SHARD=0:$N SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 $extra "$@" $P $L/unified.py fin_${TAG}_${task}_${name}_$deg --deg=$deg > $L/logs_x/${TAG}_${task}_${name}.out 2>&1 &
    g=$((g+1)); if (( g % 16 == 0 )); then wait; fi
  done < <($P $L/sched8.py $sch)
done; wait; echo ALLDONE
