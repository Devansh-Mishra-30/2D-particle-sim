#!/bin/bash
set -e

BIN=./build/sim_omp
OUT=results_omp_1t.csv
DT=0.0166667
SECONDS=10
WARMUP=200
SUBSTEPS=8

rm -f "$OUT"

export OMP_NUM_THREADS=1

for N in 1000 2000 3000 4000 5000 7000 10000 15000 20000
do
  "$BIN" --bench --n "$N" --dt "$DT" --seconds "$SECONDS" --warmup "$WARMUP" --substeps "$SUBSTEPS" --csv "$OUT"
done

echo "Wrote $OUT"
