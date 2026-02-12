CUDA Accelerated 2D Particle Simulation (CPU vs OpenMP vs CUDA)

This project is a real time 2D particle physics simulation written in C++.
It includes three backends: CPU (single thread), OpenMP (multithreaded), and CUDA (GPU).

The goal is to benchmark and visually demonstrate how parallelism changes performance at scale.

Highlights

Identical simulation logic across CPU, OpenMP, and CUDA backends

CSV based benchmarking pipeline with reproducible runs

Automatic performance plots and speedup analysis

Visual demos at 20k particles (CPU vs CUDA)

Large scale CUDA demo at 200k particles

What is included
Backends

CPU backend in cpuSim/ (builds sim_cpu)

OpenMP backend in cpuSim/ (builds sim_omp)

CUDA backend in gpuSim/ (builds particle_sim)

Benchmark Artifacts

plot_cpu_omp_cuda_ms.png
(Milliseconds per simulation step vs number of particles)

plot_cpu_omp_cuda_speedup.png
(Speedup vs CPU baseline)

portfolio_speedup_table.md

portfolio_speedup_table.csv

Key Result

At N = 20000 particles:

CPU avg evolve time: 65.97 ms

CUDA avg step time: 2.565 ms

CUDA speedup vs CPU: 25.7x

All results available in:
2D-particle-sim

Results
Frame Time Comparison

Speedup vs CPU

Demo Videos

This project includes recorded demos showing:

CPU vs CUDA at 20k particles (side-by-side comparison)

CUDA at 200k particles demonstrating large-scale GPU capability

Note: GitHub does not autoplay MP4 files inline, but it will host them for click to play or download.
If autoplay previews are desired, convert clips to GIF and embed them here.

Requirements

Linux (WSL2 works fine)

CMake

C++17 compatible compiler

SFML 3

CUDA toolkit (for gpuSim/)

Build and Run
CPU and OpenMP (cpuSim)

Build:

cd cpuSim
./build.sh


Run CPU interactively:

./build/sim_cpu --n 20000


Run OpenMP interactively (example with 8 threads):

export OMP_NUM_THREADS=8
./build/sim_omp --n 20000


Benchmark CPU:

./build/sim_cpu --bench --n 20000 --substeps 8 --seconds 10 --warmup 200 --dt 0.0166667 --csv results_cpu.csv


Benchmark OpenMP (8 threads):

export OMP_NUM_THREADS=8
./build/sim_omp --bench --n 20000 --substeps 8 --seconds 10 --warmup 200 --dt 0.0166667 --csv results_omp_8t.csv

CUDA (gpuSim)

Build:

cd gpuSim
./build.sh


Run interactively:

./build/particle_sim --n 20000


Benchmark CUDA:

./build/particle_sim --bench --n 20000 --substeps 8 --seconds 10 --warmup 200 --dt 0.0166667 --csv timings_cuda.csv

Benchmark Methodology

Fixed timestep: 1/60 s

Substeps: 8

Warmup iterations: 200

Measurement duration: 10 seconds

Metrics recorded:

Average ms per evolve step

FPS

Percentile timings

Speedup vs CPU baseline

All backends use identical simulation parameters to ensure fair comparison.

Technical Focus

This project demonstrates:

Parallel scaling characteristics

CPU single-thread limitations

OpenMP multi-thread scaling behavior

CUDA GPU acceleration benefits

Performance benchmarking discipline

Data visualization of computational scaling
