#!/bin/bash
# Real NEURON thresholds for mono / longitudinal-tripole / transverse-tripole at every
# cuff position. Sequential (each run uses all cores) -> clean config x position sweep.
# Fixed 10 um (controlled spatial selectivity; matches the validated 0.92 baseline).
set -e
PY="${PY:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SW="paper_figs/out/data/new_human_tripole_sweep"
for tag in 4x5 off15_4x5 off22_4x5 off27_4x5; do
  echo "=== NEURON thresholds: $tag ==="
  "$PY" -u paper_figs/fig5_thresholds.py "$SW/$tag" \
      --diam 10 --n-fibers 600 --trunc-mm 5 --hi-mA 50 \
      --out "$SW/$tag/thr.npz" 2>&1
  echo "--- done $tag ---"
done
echo "ALL TRIPOLE-SWEEP THRESHOLDS DONE"
