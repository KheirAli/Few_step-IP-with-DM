#!/bin/bash
# Real MRI on a second, public dataset (M4Raw, 0.3 T brain, T2): PnP-CM's own MRI pipeline with its brain consistency model, with and without the noise tilt.
cd /usr/scratch/akheirandish3/PNP; Z=xfer/m4raw/M4RawV1.5_multicoil_val.zip; O=mode_probe/overnight
for i in $(seq 1 600); do a=$(stat -c %s $Z); sleep 60; b=$(stat -c %s $Z); [ "$a" = "$b" ] && [ "$a" -gt 2900000000 ] && break; done
(cd xfer/m4raw && unzip -q -o M4RawV1.5_multicoil_val.zip) ; ls xfer/m4raw | head -3
mkdir -p PnP-CM/exp/datasets/fastMRI/M4T2 && PYTHONPATH=pylibs_m4 python3 mode_probe/m4raw_convert.py xfer/m4raw PnP-CM/exp/datasets/fastMRI/M4T2 T2 8 2>&1 | tail -2
cd PnP-CM
for a in "4 24" "8 12"; do set -- $a; sed 's|fast_mri/ema_0.9999432189950708_700000_cm_knee.pt|fast_mri/ema_0.9999432189950708_1050000_cm_brain.pt|' task_specific_args/fastmri_PD_R$1_ACS$2_gaussian1d.yml > task_specific_args/fastmri_M4T2_R$1_ACS$2_gaussian1d.yml; done
g=0; for a in "4 24" "8 12"; do set -- $a; C="--deg=fastmri --path_y=M4T2 --acc_rate=$1 --acs_lines=$2 --us_pattern=gaussian1d --config=fast_mri_320.yml --save_y"
  CUDA_VISIBLE_DEVICES=$((g%8)) ../pnpcm_env/bin/python main.py $C --exp=exp > ../mode_probe/logs_x/m4_pnp_R$1.out 2>&1 & g=$((g+1))
  CUDA_VISIBLE_DEVICES=$((g%8)) TILT=0.5 ../pnpcm_env/bin/python ../mode_probe/mri_tilt.py $C --exp=exp > ../mode_probe/logs_x/m4_tilt_R$1.out 2>&1 & g=$((g+1)); done; wait
for R in 4 8; do for v in pnp tilt; do echo "M4RAW T2, R=$R, $v: $(grep -h 'Total Average' ../mode_probe/logs_x/m4_${v}_R$R.out | sed 's/.*Total Average //' | tr '\n' ';') $(grep -h 'Error' ../mode_probe/logs_x/m4_${v}_R$R.out | tail -1 | cut -c1-150)"; done; done > ../$O/7_m4raw.txt
cat ../$O/7_m4raw.txt
