#!/bin/bash
# One worker per GPU; each takes the next line of queue_jobs.txt (under a lock) until the file is empty.
# DPS <cfg>        : DPS on cfg_paper/<cfg>.yaml          -> results_paper2/
# DAPS <task> <n>  : DAPS on a task config (noise set in the config)  -> results_paper2/<task>
# DAPSN <task>     : DAPS on a shipped task with the noise raised to 0.1 (matched to PnP-CM) -> results_paper2/<task>_n01
gpu=$1; B=/usr/scratch/akheirandish3/PNP/baselines_src; PY=/usr/scratch/akheirandish3/DAPS_project/DAPS/.venv/bin/python
while true; do
  job=$(flock $B/queue.lock bash -c "head -1 $B/queue_jobs.txt; sed -i 1d $B/queue_jobs.txt"); [ -z "$job" ] && break
  set -- $job
  if [ "$1" = DPS ]; then cd $B/diffusion-posterior-sampling; CUDA_VISIBLE_DEVICES=$gpu $PY sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=cfg_paper/$2.yaml --gpu 0 --save_dir results_paper2/$2 > log2_$2.out 2>&1
  else cd $B/DAPS; extra=""; name=$2; [ "$1" = DAPSN ] && extra="task.pixel.operator.sigma=0.1" && name=${2}_n01
    CUDA_VISIBLE_DEVICES=$gpu $PY posterior_sample.py +data=test-ffhq +model=ffhq256ddpm +task=$2 +sampler=edm_daps task_group=pixel save_dir=results_paper2 num_runs=1 sampler.diffusion_scheduler_config.num_steps=5 sampler.annealing_scheduler_config.num_steps=200 batch_size=12 data.root=dataset/paper-celeba data.start_id=0 data.end_id=107 name=$name gpu=0 save_traj=False $extra > log2_$name.out 2>&1
  fi
done
