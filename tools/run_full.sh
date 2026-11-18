#!/bin/bash
# Full BAGSFit run: all 20 scenes x 36 frames, D4 + D5, per-scene batches
# (cache-aware: safe to restart, finished samples are skipped).
cd /d/download/pcl_eval
PY="C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
EXE="D:/download/pcl_region_primitive/pcl_region_primitive/build/primitive_fit.exe"
for sc in 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019; do
  echo "=== scene $sc $(date +%H:%M:%S) ==="
  "$PY" tools/batch_test.py --samples samples/bagsfit_full \
      --results results/bagsfit_full --exe "$EXE" \
      --distance 0.02 --extra "80,12,0.05,0.01,10,0.1,5" \
      --opts "--normalK=300 --minInlierRatio=0.3" \
      --filter "${sc}_" --modes D4,D5 --dataset bagsfit 2>&1 | tail -3
done
echo "FULL_RUN_COMPLETE $(date +%H:%M:%S)"
