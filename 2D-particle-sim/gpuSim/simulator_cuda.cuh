#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "particle_data.h"

class Simulator {
public:
    Simulator(ParticleData* hostArray,
              std::size_t count,
              float boundaryRadius,
              float gravityX,
              float gravityY);

    ~Simulator();

    // Runs one simulation step on GPU.
    // If syncToHost is true, copies device particles into the provided host array for rendering.
    void update(float dt, bool syncToHost);

    // Explicit copy when you want it
    void copyToHost();

    const ParticleData* getParticlesHost() const { return h_particles; }
    std::size_t getParticleCount() const { return particleCount; }

    // Benchmark only simulation kernels (no device->host copy)
    // Returns average milliseconds per step for the GPU work inside update
    float benchmarkSteps(float dt, int warmupSteps, int measuredSteps);

private:
    void runKernels(float dt);

    std::size_t particleCount;
    float boundaryRadius;
    float gravityX, gravityY;

    ParticleData* h_particles;
    ParticleData* d_particles;
    float* d_dx;
    float* d_dy;

    cudaStream_t stream;

    cudaEvent_t evStart;
    cudaEvent_t evStop;
};
