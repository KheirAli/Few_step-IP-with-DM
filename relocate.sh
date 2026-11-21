#!/bin/bash
# The launch scripts contain the absolute path of the original server. This rewrites it to wherever this copy lives.
ROOT=$(cd "$(dirname "$0")" && pwd); OLD=/usr/scratch/akheirandish3/PNP
[ "$ROOT" = "$OLD" ] && { echo "already at the original location"; exit 0; }
grep -rlF "$OLD" "$ROOT"/mode_probe/*.sh "$ROOT"/mode_probe/*.py "$ROOT"/baselines_src/*.sh "$ROOT"/baselines_src/*.py "$ROOT"/baselines_src/*.txt 2>/dev/null | xargs -r sed -i "s|$OLD|$ROOT|g"
echo "paths rewritten to $ROOT"
