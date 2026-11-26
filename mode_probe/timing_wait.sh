#!/bin/bash
# Wait (up to 20 h) for a GPU that is truly idle (no process, <3% utilisation on three checks 20 s apart), then time every final configuration on it.
cd /usr/scratch/akheirandish3/PNP/PnP-CM; O=../mode_probe/overnight/9_timing_final.txt
for i in $(seq 1 1200); do
  for g in 0 1 2 3 4 5 6 7; do ok=1; for k in 1 2 3; do read u m <<< $(nvidia-smi -i $g --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' '); { [ "$u" -lt 3 ] && [ "$m" -lt 1500 ]; } || { ok=0; break; }; sleep 20; done
    if [ $ok = 1 ]; then echo "idle GPU $g at $(date)" > $O; TGPU=$g ../pnpcm_env/bin/python ../mode_probe/timing_final.py 2>&1 | grep -E "^\||Error" >> $O; echo "other processes on GPU $g at the end: $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l) ; utilisation now $(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader)" >> $O; cat $O; exit 0; fi; done
  sleep 60; done; echo "no idle GPU within 20 hours" > $O; cat $O
