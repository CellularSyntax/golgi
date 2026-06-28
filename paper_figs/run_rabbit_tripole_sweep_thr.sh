#!/bin/bash
# Real NEURON thresholds for mono / longitudinal-tripole / transverse-tripole at every RABBIT
# cuff position. Sequential (each run uses all cores) -> clean config x position sweep.
# Fixed 10 um (controlled spatial selectivity; matches the human fig8 sweep methodology).
set -e
PY=/opt/miniconda3/envs/fenicsx-nerve/bin/python
ROOT="/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests"
cd "$ROOT"
SW="paper_figs/out/data/rabbit_tripole_sweep"
for tag in off3_4x5 off4_4x5 off5_4x5 off6_4x5; do
  echo "=== NEURON thresholds: $tag ==="
  "$PY" -u paper_figs/fig5_thresholds.py "$SW/$tag" \
      --diam 10 --n-fibers 600 --trunc-mm 5 --hi-mA 50 \
      --out "$SW/$tag/thr.npz" 2>&1
  echo "--- done $tag ---"
done
echo "ALL RABBIT TRIPOLE-SWEEP THRESHOLDS DONE"
