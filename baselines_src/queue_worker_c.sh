#!/bin/bash
# DPS and DAPS with a CelebA-HQ diffusion prior (google/ddpm-celebahq-256) instead of the FFHQ one.  One worker per GPU.
#   DPSC <cfg>            -> diffusion-posterior-sampling/results_celeba/<cfg>
#   DAPSC <task> [n01]    -> DAPS/results_celeba/<task>[_n01]     (n01: measurement noise raised to 0.1, matched to PnP-CM)
gpu=$1; B=/usr/scratch/akheirandish3/PNP/baselines_src; PY=/usr/scratch/akheirandish3/DAPS_project/DAPS/.venv/bin/python; export CELEBA_HF=1
while true; do
  job=$(flock $B/queue_c.lock bash -c "head -1 $B/queue_jobs_c.txt; sed -i 1d $B/queue_jobs_c.txt"); [ -z "$job" ] && break
  set -- $job
  if [ "$1" = DPSC ]; then cd $B/diffusion-posterior-sampling; CUDA_VISIBLE_DEVICES=$gpu $PY sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config_fixed.yaml --task_config=cfg_paper/$2.yaml --gpu 0 --save_dir results_celeba/$2 > logc_$2.out 2>&1
  else cd $B/DAPS; extra=""; name=$2; [ "$3" = n01 ] && extra="task.pixel.operator.sigma=0.1" && name=${2}_n01
    CUDA_VISIBLE_DEVICES=$gpu $PY posterior_sample.py +data=test-ffhq +model=celebahq_hf +task=$2 +sampler=edm_daps task_group=pixel save_dir=results_celeba num_runs=1 sampler.diffusion_scheduler_config.num_steps=5 sampler.annealing_scheduler_config.num_steps=200 batch_size=12 data.root=dataset/paper-celeba data.start_id=0 data.end_id=107 name=$name gpu=0 save_traj=False $extra > logc_$name.out 2>&1
  fi
done
