#!/bin/bash
L=/usr/scratch/akheirandish3/PNP/mode_probe; T=/nethome/akheirandish3/.claude/jobs/2dbff445/tmp; cd /usr/scratch/akheirandish3/PNP/PnP-CM; P=../pnpcm_env/bin/python
MAXP=1 $P $L/bedpr_sched.py free > $T/sched_free.log 2>&1
MAXP=1 $P $L/bedpr_s8.py free > $T/s8_free.log 2>&1; MAXP=1 $P $L/bedpr_s8.py direct > $T/s8_direct.log 2>&1; touch $L/lane_b_done
bash $L/bedpr_fit.sh bedprF8
