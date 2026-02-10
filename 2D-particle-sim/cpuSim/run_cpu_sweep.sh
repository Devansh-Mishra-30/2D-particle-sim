#!/usr/bin/env bash
set -e

CSV="results_cpu.csv"
BENCH_SECONDS=10
SUBSTEPS=8
SEED=1

rm -f "$CSV"

for N in 1000 2000 3000 4000 5000 7000 10000 15000 20000; do
  ./build/simulator --bench --spawn instant --n "$N" --seconds "$BENCH_SECONDS" --substeps "$SUBSTEPS" --seed "$SEED" --csv "$CSV"
done

echo "Wrote $CSV"
