#!/bin/bash
# Clean timing of DPS and DAPS on one free GPU (waits for the 2-step run to release the GPUs).
B=/usr/scratch/akheirandish3/PNP/baselines_src; PY=/usr/scratch/akheirandish3/DAPS_project/DAPS/.venv/bin/python; L=/usr/scratch/akheirandish3/PNP/mode_probe/logs_x
while ! grep -q ALLDONE $L/launch_s2_test.log 2>/dev/null; do sleep 30; done; sleep 20
last() { tr '\r' '\n' < $1 | grep -E "[0-9]+/[0-9]+ \[" | grep -E "$2" | tail -1 | sed -E 's/.*\| *([0-9]+\/[0-9]+ \[[^]]*\]).*/\1/'; }
cd $B/diffusion-posterior-sampling
for c in super_resolution_n01 nonlinear_deblur_n01 phase_retrieval jpeg5; do
  CUDA_VISIBLE_DEVICES=0 timeout 150 $PY sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=cfg_paper/$c.yaml --gpu 0 --save_dir /tmp/dps_time_$$ > $L/time_dps_$c.out 2>&1
  echo "DPS $c (batch 1, 1000 steps): $(last $L/time_dps_$c.out '1000/1000|999/1000')"; done
rm -rf /tmp/dps_time_$$
cd $B/DAPS
for t in down_sampling nonlinear_blur phase_retrieval jpeg5; do for bs in 1 12; do
  CUDA_VISIBLE_DEVICES=0 timeout 900 $PY posterior_sample.py +data=test-ffhq +model=ffhq256ddpm +task=$t +sampler=edm_daps task_group=pixel save_dir=results_time num_runs=1 sampler.diffusion_scheduler_config.num_steps=5 sampler.annealing_scheduler_config.num_steps=200 batch_size=$bs data.root=dataset/paper-celeba data.start_id=0 data.end_id=$bs name=${t}_b$bs gpu=0 save_traj=False > $L/time_daps_${t}_$bs.out 2>&1
  echo "DAPS $t (batch $bs, 200 annealing x 5 diffusion steps): $(last $L/time_daps_${t}_$bs.out '200/200|199/200')"; done; done
rm -rf $B/DAPS/results_time; echo TIMEDONE
