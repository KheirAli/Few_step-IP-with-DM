#!/bin/bash
# usage: [GPULIST="0 1 .."] [MINFREE=7000] gpurun.sh <logfile> <command...>
# Runs the command on the allowed GPU with the most free memory (at least MINFREE MiB), and runs it again if it dies with a GPU memory error.
log=$1; shift
for try in $(seq 1 60); do
  sleep $((RANDOM % 10))
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' -v m=${MINFREE:-7000} -v allow=" ${GPULIST:-0 1 2 3 4 5 6 7} " '{f=$3-$2; if (index(allow," "$1" ") && f>m && f>best){best=f; b=$1}} END{print b}')
  if [ -z "$g" ]; then sleep 60; continue; fi
  CUDA_VISIBLE_DEVICES=$g "$@" > $log 2>&1
  grep -q "OutOfMemoryError\|CUDA out of memory\|CUDA error" $log || exit 0
  sleep 30
done; echo "GPURUN gave up" >> $log; exit 1
