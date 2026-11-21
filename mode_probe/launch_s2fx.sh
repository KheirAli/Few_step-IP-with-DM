#!/bin/bash
# Two steps (schedule from s2_best.json) with the frequency-weighted data-fit.  SET=sel (24 non-test images) or test (107).  GPUs 1-7 only.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; SET=${SET:-sel}
if [ $SET = sel ]; then D=$(realpath ../mode_probe/data100); SH="0:24"; else D=$(realpath ../mode_probe/data_paper); SH="0:54 54:107"; fi
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1"
  [jpeg]="jpeg jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg jpeg" [deblur_nl]="deblur_nl deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval inpainting INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl deblur_nl NEWOP=quant" [onebit]="deblur_nl deblur_nl NEWOP=onebit" )
declare -A VAR=( [fw05]="FW=1 FWP=0.5" [fw05m]="FW=1 FWP=0.5 FWMIN=0.4" [fw05k12]="FW=1 FWP=0.5 FWMIN=0.4 FWK=12" [rx2]="RHOX=2" [rx05]="RHOX=0.5" )
g=0
for task in ${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit}; do set -- ${CFG[$task]}; deg=$1; sch=$2; shift 2
  want=$($P -c "import json;print(json.load(open('$L/s2_best.json'))['$task'])"); line=$($P $L/sched2.py $sch | grep "^$want "); a=($line)
  for v in ${VARS:-fw05 fw05m}; do for sh in $SH; do
    env CUDA_VISIBLE_DEVICES=$((1 + g%7)) DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 SIGS=${a[1]} SIGPOS=${a[2]} "$@" ${VAR[$v]} $EXTRA $P $L/unified.py fin_s2${SET}${v}_${task}_${want}_$deg --deg=$deg > $L/logs_x/s2${SET}${v}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1))
    if (( g % 14 == 0 )); then wait; fi; done; done
done; wait; echo ALLDONE
