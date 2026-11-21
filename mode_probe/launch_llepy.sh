#!/bin/bash
# stub: the real run was started early as launch_llepy_now.sh; wait for it
while ! grep -q ALLDONE /usr/scratch/akheirandish3/PNP/mode_probe/logs_x/launch_llepy_now.log 2>/dev/null; do sleep 60; done; echo ALLDONE
