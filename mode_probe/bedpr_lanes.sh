#!/bin/bash
# remaining bedroom pipeline in three single-process lanes (never more than 3 GPU processes)
L=/usr/scratch/akheirandish3/PNP/mode_probe; T=/nethome/akheirandish3/.claude/jobs/2dbff445/tmp; cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python
wf() { until grep -q "$1" $2 2>/dev/null; do sleep 15; done; }; we() { until [ -e $1 ]; do sleep 15; done; }
( wf "PREFILL DONE" $T/pre_free_43.log; bash $L/bedpr_prefill.sh net 43:86 > $T/pre_net_43.log 2>&1; bash $L/bedpr_prefill.sh net 86:128 > $T/pre_net_86.log 2>&1
  we $L/bedpr_sched_direct.json; bash $L/bedpr_fit.sh bedprD4; we $L/lane_b_done; bash $L/bedpr_fit.sh bedprD8 ) > $T/lane_a.log 2>&1 &
( we $L/bedpr_sched_direct.json; wf "PREFILL DONE" $T/pre_free_43.log; echo 1 > $L/bedpr_maxp_sched.txt; MAXP=1 $P $L/bedpr_sched.py free > $T/sched_free.log 2>&1
  MAXP=1 $P $L/bedpr_s8.py free > $T/s8_free.log 2>&1; MAXP=1 $P $L/bedpr_s8.py direct > $T/s8_direct.log 2>&1; touch $L/lane_b_done
  bash $L/bedpr_fit.sh bedprF8 ) > $T/lane_b.log 2>&1 &
( wf "PREFILL DONE" $T/pre_net_0.log; we $L/bedpr_sched_free.json; bash $L/bedpr_fit.sh bedprF4; wf "PREFILL DONE" $T/pre_net_86.log; bash $L/bedpr_fit.sh bedprN4
  we $L/lane_b_done; bash $L/bedpr_fit.sh bedprN8 ) > $T/lane_c.log 2>&1 &
wait; echo LANES ALLDONE
