#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <output.csv> <threads>"
  echo "Example: $0 results_omp_8t.csv 8"
  exit 1
fi

CSV="$1"
THREADS="$2"

if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: threads must be a positive integer. Got: '$THREADS'"
  exit 1
fi

export OMP_NUM_THREADS="$THREADS"

SEED=1
SUBSTEPS=8
BENCH_SECONDS=10

rm -f "$CSV"

for N in 1000 2000 3000 4000 5000 7000 10000 15000 20000; do
  ./build/sim_omp --bench --spawn instant --n "$N" \
    --seconds "$BENCH_SECONDS" --substeps "$SUBSTEPS" \
    --seed "$SEED" --csv "$CSV"
done

echo "Wrote $CSV"
