#!/bin/bash
# Real MRI, exact row-wise data step: PnP-CM's penalties were set for 10 solver iterations (an early-stopped solve), so the penalty is rescaled.
# The factor is chosen on GENERATED slices (PSNR after the 3 calibration rounds); only the chosen one is run on the 40 test slices.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; O=$L/overnight
one() { R=$1; ACS=$2; g=$3; best=""; bp=0; A="--deg=fastmri --acc_rate=$R --acs_lines=$ACS --us_pattern=gaussian1d --config=fast_mri_320.yml --exp=exp"
  for X in 1 3 10 30; do W=$(realpath $L/lle)/mri_m4x${X}_R$R.pt; rm -f $W
    for K in 1 2 3; do LG=$L/mri_callogx_R$R; rm -rf $LG
      CUDA_VISIBLE_DEVICES=$g MRIEXACT=1 MRIRHOX=$X MRIMODE=cal MRIK=$K MRILOG=$LG MRIW=$W TILT=0.5 $P $L/mri_ours.py $A --path_y=M4GEN > $L/logs_x/mrix2_cal_R${R}_${X}_$K.out 2>&1
      MRIMODE=fit MRIK=$K MRILOG=$LG MRIW=$W $P $L/mri_ours.py > /dev/null; done; rm -rf $L/mri_callogx_R$R
    p=$(grep -h 'Total Average PSNR' $L/logs_x/mrix2_cal_R${R}_${X}_3.out | sed 's/.*: //; s/ .*//'); echo "generated slices, R=$R, exact step, penalty x$X: PSNR ${p:-failed}"
    [ -n "$p" ] && [ "$(echo "$p > $bp" | bc)" = 1 ] && { bp=$p; best=$X; }; done
  W=$(realpath $L/lle)/mri_m4x${best}_R$R.pt
  CUDA_VISIBLE_DEVICES=$g MRIEXACT=1 MRIRHOX=$best MRIMODE=use MRIW=$W TILT=0.5 $P $L/mri_ours.py $A --path_y=M4T2 > $L/logs_x/mrix2_test_R$R.out 2>&1
  echo "M4RAW T2, R=$R, ours with the EXACT row-wise data step, penalty x$best (chosen on generated slices): $(grep -h 'Total Average' $L/logs_x/mrix2_test_R$R.out | sed 's/.*Total Average //' | tr '\n' ';') $(grep -h Error $L/logs_x/mrix2_test_R$R.out | tail -1 | cut -c1-120) | time per slice: $(tr '\r' '\n' < $L/logs_x/mrix2_test_R$R.out | grep -o '[0-9.]*s/it\|[0-9.]*it/s' | tail -1)"; }
one 4 24 ${G1:-4} > $O/8_mri_exact2_R4.txt 2>&1 & one 8 12 ${G2:-0} > $O/8_mri_exact2_R8.txt 2>&1 & wait; cat $O/8_mri_exact2_R4.txt $O/8_mri_exact2_R8.txt
