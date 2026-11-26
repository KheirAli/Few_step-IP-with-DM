#!/bin/bash
# Our pipeline on real MRI (M4Raw T2, 40 slices): generated calibration images (coil maps of 4 other subjects), in-loop band combination fitted step by step, noise tilt.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; O=$L/overnight; mkdir -p $O
for d in M4T2cal M4GEN; do for a in "4 24" "8 12"; do set -- $a; cp task_specific_args/fastmri_M4T2_R$1_ACS$2_gaussian1d.yml task_specific_args/fastmri_${d}_R$1_ACS$2_gaussian1d.yml; done; done
rm -rf exp/datasets/fastMRI/M4GEN
CUDA_VISIBLE_DEVICES=${G1:-4} MRIMODE=gen MRIGEN=M4GEN MRINGEN=4 $P $L/mri_ours.py --deg=fastmri --path_y=M4T2cal --acc_rate=4 --acs_lines=24 --us_pattern=gaussian1d --config=fast_mri_320.yml --exp=exp > $L/logs_x/mriours_gen.out 2>&1
echo "generated calibration slices: $(ls exp/datasets/fastMRI/M4GEN | wc -l)"
one() { R=$1; ACS=$2; g=$3; W=$(realpath $L/lle)/mri_m4_R$R.pt; rm -f $W; A="--deg=fastmri --acc_rate=$R --acs_lines=$ACS --us_pattern=gaussian1d --config=fast_mri_320.yml --exp=exp"
  for K in 1 2 3; do LG=$L/mri_callog_R$R; rm -rf $LG
    CUDA_VISIBLE_DEVICES=$g MRIMODE=cal MRIK=$K MRILOG=$LG MRIW=$W TILT=0.5 $P $L/mri_ours.py $A --path_y=M4GEN > $L/logs_x/mriours_cal_R${R}_$K.out 2>&1
    MRIMODE=fit MRIK=$K MRILOG=$LG MRIW=$W $P $L/mri_ours.py; done; rm -rf $L/mri_callog_R$R
  echo "generated images, R=$R, ours: $(grep -h 'Total Average' $L/logs_x/mriours_cal_R${R}_3.out | sed 's/.*Total Average //' | tr '\n' ';') (combination of the last step not yet applied)"
  CUDA_VISIBLE_DEVICES=$g MRIMODE=use MRIW=$W TILT=0.5 $P $L/mri_ours.py $A --path_y=M4T2 > $L/logs_x/mriours_test_R$R.out 2>&1
  echo "M4RAW T2, R=$R, ours (tilt + band combination): $(grep -h 'Total Average' $L/logs_x/mriours_test_R$R.out | sed 's/.*Total Average //' | tr '\n' ';') $(grep -h Error $L/logs_x/mriours_test_R$R.out | tail -1 | cut -c1-160)"; }
one 4 24 ${G1:-4} > $O/8_mri_ours_R4.txt 2>&1 & one 8 12 ${G2:-5} > $O/8_mri_ours_R8.txt 2>&1 & wait; cat $O/8_mri_ours_R4.txt $O/8_mri_ours_R8.txt
