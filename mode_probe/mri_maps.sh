#!/bin/bash
# Real MRI: two more data steps.  usage: mri_maps.sh <pattern gaussian1d|regular1d> <R> <ACS> <data step grappa|sense>
# The penalty factor is chosen on generated slices; the 40 test slices are run once.  For the regular pattern PnP-CM is run on the same mask first.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python; L=../mode_probe; PT=$1; R=$2; ACS=$3; D=$4; T=${PT}_${D}_R$R${SFX}; best=""; bp=0; A="--deg=fastmri --acc_rate=$R --acs_lines=$ACS --us_pattern=$PT --config=fast_mri_320.yml --exp=exp"; export MINFREE=9000
if [ $PT = regular1d ] && [ -z "$XS" ]; then bash $L/gpurun.sh $L/logs_x/mrimap_${T}_pnp.out env MRIDATA=cg MRIMODE=use MRIW= TILT=0 $P $L/mri_ours.py $A --path_y=M4T2
  echo "M4RAW T2, $PT R=$R, PnP-CM (10 solver iterations, no tilt, no combination): $(grep -h 'Total Average' $L/logs_x/mrimap_${T}_pnp.out | sed 's/.*Total Average //' | tr '\n' ';') data steps $(grep -oh "this slice [0-9.]*" $L/logs_x/mrimap_${T}_pnp.out | awk '{s+=$3; n++} END{printf "%.1f ms", s/n}')"; fi
for X in ${XS:-1 3 10}; do W=$(realpath $L/lle)/mri_${T}_x$X.pt; rm -f $W
  for K in 1 2 3; do LG=$L/mri_callog_$T; rm -rf $LG
    bash $L/gpurun.sh $L/logs_x/mrimap_${T}_cal_${X}_$K.out env MRIDATA=$D MRIRHOX=$X MRIMODE=cal MRIK=$K MRILOG=$LG MRIW=$W TILT=0.5 $P $L/mri_ours.py $A --path_y=${GEN:-M4GEN}
    MRIMODE=fit MRIK=$K MRILOG=$LG MRIW=$W $P $L/mri_ours.py > /dev/null; done; rm -rf $LG
  p=$(grep -h 'Total Average PSNR' $L/logs_x/mrimap_${T}_cal_${X}_3.out | sed 's/.*: //; s/ .*//'); echo "generated slices, $PT R=$R, $D step, penalty x$X: PSNR ${p:-failed}"
  [ -n "$p" ] && [ "$(echo "$p > $bp" | bc)" = 1 ] && { bp=$p; best=$X; }; done
bash $L/gpurun.sh $L/logs_x/mrimap_${T}_test.out env MRIDATA=$D MRIRHOX=$best MRIMODE=use MRIW=$(realpath $L/lle)/mri_${T}_x$best.pt TILT=0.5 $P $L/mri_ours.py $A --path_y=M4T2
echo "M4RAW T2, $PT R=$R, ours with the $D data step, penalty x$best: $(grep -h 'Total Average' $L/logs_x/mrimap_${T}_test.out | sed 's/.*Total Average //' | tr '\n' ';') data steps $(grep -oh "this slice [0-9.]*" $L/logs_x/mrimap_${T}_test.out | awk '{s+=$3; n++} END{printf "%.1f ms", s/n}') | $(grep -oh 'MRISENSE.*' $L/logs_x/mrimap_${T}_test.out | head -1) $(grep -oh "MRIGRAPPA lines left empty [0-9]* of [0-9]* missing.*" $L/logs_x/mrimap_${T}_test.out | awk '{e+=$5; m=$7; p+=$NF; n++} END{if (n) printf "lines left empty %.1f of %d missing on average; interpolation alone PSNR %.2f", e/n, m, p/n}') $(grep -h Error $L/logs_x/mrimap_${T}_test.out | tail -1 | cut -c1-120)"
