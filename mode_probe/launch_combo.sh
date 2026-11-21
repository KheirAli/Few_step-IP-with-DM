#!/bin/bash
# Same budget (4 model calls), spent differently.  SET=sel (24 non-test images) or test (107).  Base method, no per-task fixes.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; SET=${SET:-sel}; g=0
if [ $SET = sel ]; then D=$(realpath $L/data100); SH="0:24"; else D=$(realpath $L/data_paper); SH="0:54 54:107"; fi
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1"
  [jpeg]="jpeg jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg jpeg" [deblur_nl]="deblur_nl deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval inpainting INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl deblur_nl NEWOP=quant" [onebit]="deblur_nl deblur_nl NEWOP=onebit" )
sig() { $P - "$1" "$2" <<'EOF'
import sys, numpy as np
SHIP={"sr_bicubic":[0.465,0.245,0.01,0.01],"deblur_gauss":[0.327,0.132,0.01,0.01],"inpainting":[0.327,0.132,0.01,0.01],"jpeg":[0.694,0.571,0.351,0.01],"deblur_nl":[0.694,0.635,0.56,0.459]}
s=SHIP[sys.argv[1]]; pos=[float(v) for v in sys.argv[2].split(",")]
def at(p):
    i=int(np.floor(min(p,2.999))); t=p-i; return float(np.exp((1-t)*np.log(s[i])+t*np.log(s[i+1])))
print(",".join("%.4f"%at(p) for p in pos))
EOF
}
for task in ${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit}; do c=(${CFG[$task]}); deg=${c[0]}; sch=${c[1]}; cfg="${c[@]:2}"
  w2=$($P -c "import json;print(json.load(open('$L/s2_best.json'))['$task'])"); a2=($($P $L/sched2.py $sch | grep "^$w2 "))
  for v in ${VARS:-avg avgfit fork013 fork012 search013 search023}; do
    case $v in avg) ex="COMBO=avg SIGS=${a2[1]} SIGPOS=${a2[2]}";; avgfit) ex="COMBO=avgfit SIGS=${a2[1]} SIGPOS=${a2[2]}";;
      fork013) ex="COMBO=fork SIGS=$(sig $sch 0,1,3) SIGPOS=0,1,3";; fork012) ex="COMBO=fork SIGS=$(sig $sch 0,1,2) SIGPOS=0,1,2";;
      search013) ex="COMBO=search SIGS=$(sig $sch 0,1,3) SIGPOS=0,1,3";; search023) ex="COMBO=search SIGS=$(sig $sch 0,2,3) SIGPOS=0,2,3";; esac
    for sh in $SH; do env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 $cfg $ex $P $L/unified.py fin_cb${SET}_${v}_${task}_$deg --deg=$deg > $L/logs_x/cb${SET}_${v}_${task}_${sh%%:*}.out 2>&1 &
      g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; done; done; done; wait; echo ALLDONE
