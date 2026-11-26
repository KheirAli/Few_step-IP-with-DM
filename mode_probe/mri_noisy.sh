#!/bin/bash
# Real MRI, selection and weights on generated slices WITH measurement noise at the level of the real scans (estimated on the 4 calibration subjects).
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; O=$L/overnight; export GEN=M4GENN SFX=n
rm -rf exp/datasets/fastMRI/M4GENN
MINFREE=9000 bash $L/gpurun.sh $L/logs_x/mrinoisy_gen.out env MRIMODE=gen MRIGEN=M4GENN MRINGEN=4 MRINOISE=1 $P $L/mri_ours.py --deg=fastmri --path_y=M4T2cal --acc_rate=4 --acs_lines=24 --us_pattern=gaussian1d --config=fast_mri_320.yml --exp=exp
echo "noisy generated slices: $(ls exp/datasets/fastMRI/M4GENN | wc -l); noise std per k-space sample, mean over calibration slices: $(grep -oh 'MRIGEN noise std per k-space sample [0-9.]*' $L/logs_x/mrinoisy_gen.out | awk '{s+=$NF; n++} END{printf "%.4f", s/n}')" > $O/12_noise.txt
g() { R=$1; ACS=$2; XS="1" bash $L/mri_maps.sh gaussian1d $R $ACS cg; XS="1 3 10 30" bash $L/mri_maps.sh gaussian1d $R $ACS exact; XS="3 10 30 100" bash $L/mri_maps.sh gaussian1d $R $ACS grappa; }
g 4 24 > $O/12_g4.txt 2>&1 & g 8 12 > $O/12_g8.txt 2>&1 &
XS="0.3 1 3 10" bash $L/mri_maps.sh regular1d 4 24 sense > $O/12_s4.txt 2>&1 & XS="0.3 1 3 10" bash $L/mri_maps.sh regular1d 8 12 sense > $O/12_s8.txt 2>&1 & wait
cat $O/12_noise.txt $O/12_g4.txt $O/12_g8.txt $O/12_s4.txt $O/12_s8.txt | sed 's/ data steps.*//'
