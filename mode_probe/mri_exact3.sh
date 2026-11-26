#!/bin/bash
# Real MRI, exact row-wise step, penalty factor chosen on generated slices - memory-safe rerun for one undersampling rate.  usage: mri_exact3.sh <R> <ACS>
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; R=$1; ACS=$2; best=""; bp=0; A="--deg=fastmri --acc_rate=$R --acs_lines=$ACS --us_pattern=gaussian1d --config=fast_mri_320.yml --exp=exp"
for X in 1 3 10; do W=$(realpath $L/lle)/mri_m4y${X}_R$R.pt; rm -f $W
  for K in 1 2 3; do LG=$L/mri_callogy_R$R; rm -rf $LG
    MINFREE=9000 bash $L/gpurun.sh $L/logs_x/mrix3_cal_R${R}_${X}_$K.out env MRIEXACT=1 MRIRHOX=$X MRIMODE=cal MRIK=$K MRILOG=$LG MRIW=$W TILT=0.5 $P $L/mri_ours.py $A --path_y=M4GEN
    MRIMODE=fit MRIK=$K MRILOG=$LG MRIW=$W $P $L/mri_ours.py | sed 's/ |.*//' | cut -c1-40; done; rm -rf $LG
  p=$(grep -h 'Total Average PSNR' $L/logs_x/mrix3_cal_R${R}_${X}_3.out | sed 's/.*: //; s/ .*//'); echo "generated slices, R=$R, exact step, penalty x$X: PSNR ${p:-failed}"
  [ -n "$p" ] && [ "$(echo "$p > $bp" | bc)" = 1 ] && { bp=$p; best=$X; }; done
W=$(realpath $L/lle)/mri_m4y${best}_R$R.pt
MINFREE=9000 bash $L/gpurun.sh $L/logs_x/mrix3_test_R$R.out env MRIEXACT=1 MRIRHOX=$best MRIMODE=use MRIW=$W TILT=0.5 $P $L/mri_ours.py $A --path_y=M4T2
echo "M4RAW T2, R=$R, ours with the EXACT row-wise data step, penalty x$best (chosen on generated slices): $(grep -h 'Total Average' $L/logs_x/mrix3_test_R$R.out | sed 's/.*Total Average //' | tr '\n' ';')"
