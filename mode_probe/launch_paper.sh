#!/bin/bash
# 2026-11-18: PnP-CM baseline and our closed-form method on the validation images published with the PnP-CM paper.
# Usage: N=<number of images> ./launch_paper.sh      (one job per GPU)
cd /usr/scratch/akheirandish3/PNP/PnP-CM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=$(realpath ../mode_probe/data_paper); P=../pnpcm_env/bin/python; L=../mode_probe; G=(${GPUS:-0 1 2 3 4 5 6 7}); n=${#G[@]}; N=${N:-100}; S=${S:-25}
jobs_=()
add(){ for ((a=${FROM:-0}; a<N; a+=S)); do b=$((a+S)); [ $b -gt $N ] && b=$N; jobs_+=("$a:$b $*"); done; }
for deg in sr_bicubic deblur_gauss inpainting; do add pvours $deg MODE=unified LINEXACT=1 TILT=0.5; done
add pvours deblur_nl MODE=unified WARMX=lin NLKER=1 MEC=step TILT=0.5
add pvours jpeg MODE=unified JPROJ=1 JSOFT=1 JCOLD=1 TILT=0.5
add pvours phase_retrieval MODE=unified INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1 TILT=0.5
for deg in sr_bicubic deblur_gauss inpainting deblur_nl jpeg phase_retrieval; do add pvbase $deg MODE=base; done
lane(){ l=$1; for ((i=l; i<${#jobs_[@]}; i+=n)); do set -- ${jobs_[$i]}; sh=$1; v=$2; deg=$3; shift 3
  env CUDA_VISIBLE_DEVICES=${G[$l]} DATA=$D SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 "$@" $P $L/unified.py fin_${v}_$deg --deg=$deg > $L/logs_x/${v}_${deg}_s${sh%%:*}.out 2>&1
done; }
for l in $(seq 0 $((n-1))); do lane $l & done; wait; echo ALLDONE
