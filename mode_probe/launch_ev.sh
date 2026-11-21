#!/bin/bash
# Items 1 and 2: expected-value data-fits and a calibrated error model, at 4 and 2 steps.
#   calibration images: data100 24:48     selection images: data100 0:24     test: the 107 paper images
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; g=0
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1"
  [jpeg]="jpeg jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg jpeg" [deblur_nl]="deblur_nl deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval inpainting INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl deblur_nl NEWOP=quant" [onebit]="deblur_nl deblur_nl NEWOP=onebit" )
ALL="sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit"; EVT="jpeg hdr quant onebit"
run() { # name set steps task extra...
  local name=$1 set=$2 st=$3 task=$4; shift 4; local c=(${CFG[$task]}); local deg=${c[0]} sch=${c[1]}; local cfg="${c[@]:2}"; local sg=""
  if [ $st = 2 ]; then local want=$($P -c "import json;print(json.load(open('$L/s2_best.json'))['$task'])"); local a=($($P $L/sched2.py $sch | grep "^$want ")); sg="SIGS=${a[1]} SIGPOS=${a[2]}"; fi
  local D SH; case $set in cal) D=$L/data100; SH="24:48";; sel) D=$L/data100; SH="0:24";; test) D=$L/data_paper; SH="0:54 54:107";; esac
  for sh in $SH; do
    env CUDA_VISIBLE_DEVICES=$((1 + g%7)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 $cfg $sg "$@" $P $L/unified.py fin_ev${set}${st}_${name}_${task}_$deg --deg=$deg > $L/logs_x/ev${set}${st}_${name}_${task}_${sh%%:*}.out 2>&1 &
    g=$((g+1)); if (( g % 14 == 0 )); then wait; fi; done; }
ev() { [ $1 = onebit ] && echo "OBMEAN=cal" || echo "EVFIT=1"; }
STAGE=${STAGE:-all}
if [[ $STAGE = all || $STAGE = cal ]]; then
  for st in 4 2; do for t in $ALL; do run calbase cal $st $t CALW=c${st}_base_$t; done; done; wait
  for r in 1 2 3; do prev=$([ $r = 1 ] && echo base || echo ev$((r-1)))
    for st in 4 2; do for t in $EVT; do run calev$r cal $st $t $(ev $t) CALU=c${st}_${prev}_$t CALW=c${st}_ev${r}_$t; done; done; wait; done
  for st in 4 2; do for t in $ALL; do run calfw cal $st $t FW=cal CALU=c${st}_base_$t CALW=c${st}_fw1_$t; done; done; wait
fi
if [[ $STAGE = all || $STAGE = sel ]]; then
  for st in 4 2; do
    for t in $EVT; do run ev sel $st $t $(ev $t) CALU=c${st}_ev3_$t; run evfw sel $st $t $(ev $t) FW=cal CALU=c${st}_ev3_$t; done
    for t in $ALL; do run fwcal sel $st $t FW=cal CALU=c${st}_base_$t; run fwcal2 sel $st $t FW=cal CALU=c${st}_fw1_$t; run fwfull sel $st $t FW=calfull CALU=c${st}_base_$t; done
  done; wait
fi
if [[ $STAGE = test ]]; then for spec in $TESTSPEC; do IFS=: read st t v <<< "$spec"
    case $v in ev) run ev test $st $t $(ev $t) CALU=c${st}_ev3_$t;; evfw) run evfw test $st $t $(ev $t) FW=cal CALU=c${st}_ev3_$t;; fwcal) run fwcal test $st $t FW=cal CALU=c${st}_base_$t;; fwcal2) run fwcal2 test $st $t FW=cal CALU=c${st}_fw1_$t;; fwfull) run fwfull test $st $t FW=calfull CALU=c${st}_base_$t;; esac; done; wait; fi
echo ALLDONE
