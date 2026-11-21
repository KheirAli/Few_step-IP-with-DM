#!/bin/bash
# SERVER-1 JOB: the final 4-call method on the 107 test images with 3 sampler seeds (seed 1 must reproduce the reference table).
# env: NGPU (default 8), PAR (jobs at a time, default 2*NGPU), SEEDS ("1 2 3"), TASKS, SHS ("0:54 54:107")
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=../pnpcm_env/bin/python; L=../mode_probe; mkdir -p $L/logs_x; NGPU=${NGPU:-8}; PAR=${PAR:-$((2*NGPU))}
declare -A CFG=( [sr_bicubic]="sr_bicubic LINEXACT=1" [deblur_gauss]="deblur_gauss LINEXACT=1" [inpainting]="inpainting LINEXACT=1 FW=1 FWP=0.5" [jpeg]="jpeg JPROJ=1 JSOFT=1 JCOLD=1 FW=1 FWP=0.5 FWSTEPS=1,2,3"
  [deblur_nl]="deblur_nl WARMX=lin NLKER=1 MEC=step FW=1 FWP=0.5 FWMIN=0.4" [prmask]="phase_retrieval INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=none LINEXACT=1 FW=1 FWP=0.5"
  [hdr]="deblur_nl NEWOP=hdr HDRX=1 RHOX=2" [quant]="deblur_nl NEWOP=quant FW=1 FWP=0.5 FWMIN=0.4" [onebit]="deblur_nl NEWOP=onebit OBMEAN=1.0,0.5,0.3,0.2 FW=1 FWP=0.5 FWMIN=0.4" )
declare -A W=( [sr_bicubic]=sr_bicubic [deblur_gauss]=deblur_gauss [inpainting]=best_inpainting [jpeg]=best_jpeg [deblur_nl]=best_deblur_nl [prmask]=best_prmask [hdr]=best_hdr [quant]=best_quant [onebit]=best_onebit )
g=0; for seed in ${SEEDS:-1 2 3}; do for task in ${TASKS:-sr_bicubic deblur_gauss inpainting jpeg deblur_nl hdr quant onebit prmask}; do c=(${CFG[$task]}); deg=${c[0]}; for sh in ${SHS:-0:54 54:107}; do
  env CUDA_VISIBLE_DEVICES=$((g%NGPU)) DATA=$(realpath $L/data_paper) SHARD=$sh SAMPLE_SEED=$seed RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 ${c[@]:1} LLE=$(realpath $L/lle)/${W[$task]}.json \
    $P $L/unified.py fin_seed${seed}_${task}_$deg --deg=$deg > $L/logs_x/seed${seed}_${task}_${sh%%:*}.out 2>&1 & g=$((g+1)); if (( g % PAR == 0 )); then wait; fi; done; done; done; wait; echo ALLDONE
