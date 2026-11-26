#!/bin/bash
# print the id of the GPU with the most free memory (at least ${1:-8000} MiB); wait until one exists.  Optional $2: space-separated ids to avoid.
need=${1:-8000}; avoid=" ${2:-} "
while true; do
  best=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' -v need=$need -v avoid="$avoid" '{f=$3-$2; if (f>=need && index(avoid," "$1" ")==0 && f>m) {m=f; b=$1}} END {print b}')
  [ -n "$best" ] && { echo $best; exit 0; }; sleep 30
done
