#!/bin/bash
set -e

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
echo "Built: build/particle_sim"
