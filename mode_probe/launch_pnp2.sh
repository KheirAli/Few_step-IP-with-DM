#!/bin/bash
# PnP-CM's own data-fit solver inside the same loop, with white noise and the repository's start, on OUR schedules.
#   STAGE=val : shipped 4-step schedule on the test images (must reproduce the MODE=base numbers)
#   STAGE=sel : all eight 2-step candidates on 24 non-test images     STAGE=test : 2 steps on the 107 test images (our choice and its own best)
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; g=0; STAGE=${STAGE:-sel}
declare -A CFG=( [sr_bicubic]="sr_bicubic sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss deblur_gauss LINEXACT=1" [inpainting]="inpainting inpainting LINEXACT=1"
  [jpeg]="jpeg jpeg SOLVER=repo" [deblur_nl]="deblur_nl deblur_nl SOLVER=repo" [hdr]="deblur_nl deblur_nl NEWOP=hdr SOLVER=repo" [quant]="deblur_nl deblur_nl NEWOP=quant SOLVER=repo" [onebit]="deblur_nl deblur_nl NEWOP=onebit SOLVER=repo" )
run() { # tag task shard-set extra
  local tag=$1 task=$2 set=$3; shift 3; local c=(${CFG[$task]}); local deg=${c[0]}; local cfg="${c[@]:2}"; local D SH
  case $set in sel) D=$L/data100; SH="0:24";; test) D=$L/data_paper; SH="0:54 54:107";; esac
  for sh in $SH; do env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$(realpath $D) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0 $cfg "$@" $P $L/unified.py fin_pp${set}_${tag}_${task}_$deg --deg=$deg > $L/logs_x/pp${set}_${tag}_${task}_${sh%%:*}.out 2>&1 &
    g=$((g+1)); if (( g % 16 == 0 )); then wait; fi; done; }
TASKS=${TASKS:-sr_bicubic deblur_gauss inpainting jpeg deblur_nl hdr quant onebit}
for task in $TASKS; do c=(${CFG[$task]}); sch=${c[1]}
  if [ $STAGE = val ]; then run ship4 $task test; fi
  if [ $STAGE = sel ]; then while read name sig pos; do run $name $task sel SIGS=$sig SIGPOS=$pos; done < <($P $L/sched2.py $sch); fi
  if [ $STAGE = test ]; then for which in ours own; do f=$([ $which = ours ] && echo s2_best.json || echo pp2_best.json); want=$($P -c "import json;print(json.load(open('$L/$f'))['$task'])")
      a=($($P $L/sched2.py $sch | grep "^$want ")); run 2$which $task test SIGS=${a[1]} SIGPOS=${a[2]}; done; fi
done; wait; echo ALLDONE
