#!/bin/bash
set -e

BIN=./build/particle_sim
OUT=timings_cuda.csv
DT=0.0166667
WARMUP=200

rm -f $OUT

$BIN --bench --n 1000    --warmup $WARMUP --steps 2000 --dt $DT --csv $OUT
$BIN --bench --n 5000    --warmup $WARMUP --steps 2000 --dt $DT --csv $OUT
$BIN --bench --n 20000   --warmup $WARMUP --steps 1000 --dt $DT --csv $OUT
$BIN --bench --n 100000  --warmup $WARMUP --steps 300  --dt $DT --csv $OUT

cat $OUT
