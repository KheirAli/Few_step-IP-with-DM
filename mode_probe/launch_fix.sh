#!/bin/bash
# Three fixes on all tasks, 4 steps.  SET=sel -> 24 non-test images (data100), SET=test -> the 107 test images.
#   fw05 / fw10 : frequency-weighted data-fit (less trust in the model's image at low frequencies), power 0.5 / 1.0
#   ft15 / ft20 : last model call told a noise level 1.5x / 2x higher (cleaner final image)
#   lt1         : true operator in the last data-fit (only where the method uses a surrogate or a reduction)
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; SET=${SET:-sel}
if [ $SET = sel ]; then D=$(realpath ../mode_probe/data100); SH="0:24"; else D=$(realpath ../mode_probe/data_paper); SH="0:54 54:107"; fi
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1"
  [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1" [jpegtrue]="jpeg" [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step"
  [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1"
  [hdr]="deblur_nl NEWOP=hdr HDRX=1" [quant]="deblur_nl NEWOP=quant" [onebit]="deblur_nl NEWOP=onebit" )
declare -A VAR=( [base]="" [fw05]="FW=1 FWP=0.5" [fw10]="FW=1 FWP=1.0" [ft15]="FINTOLD=1.5" [ft20]="FINTOLD=2.0" [lt1]="LASTTRUE=1" [rx05]="RHOX=0.5" [rx025]="RHOX=0.25" [rx2]="RHOX=2" [fw025]="FW=1 FWP=0.25" [fw05c]="FW=1 FWP=0.5 FWC=0.1" [fw10c]="FW=1 FWP=1.0 FWC=0.06" [fw05m]="FW=1 FWP=0.5 FWMIN=0.4" )
g=0
for task in ${TASKS:-sr_bicubic deblur_gauss inpainting jpeg jpegtrue deblur_nl prmask hdr quant onebit}; do set -- ${CFG[$task]}; deg=$1; shift
  for v in ${VARS:-base fw05 fw10 ft15 ft20 lt1}; do
    if [ $v = lt1 ] && [[ ! " jpeg deblur_nl prmask hdr " =~ " $task " ]]; then continue; fi
    for sh in $SH; do
      env CUDA_VISIBLE_DEVICES=$((g%8)) DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 "$@" ${VAR[$v]} $P $L/unified.py fin_fx${SET}_${v}_${task}_$deg --deg=$deg > $L/logs_x/fx${SET}_${v}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1))
      if (( g % 16 == 0 )); then wait; fi; done; done
done; wait; echo ALLDONE
