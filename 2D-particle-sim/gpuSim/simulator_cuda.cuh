// simulator_cuda.cuh
#pragma once
#include <cuda_runtime.h>    // CUDA runtime API
#include "particle_data.h"

class Simulator {
public:
    // Constructor: allocate GPU memory and initialize particle data
    Simulator(ParticleData* hostArray, std::size_t count, float boundaryRadius,
              float gravityX, float gravityY);
    // Destructor: free GPU memory
    ~Simulator();
    // Advance the simulation by a time step (dt seconds)
    void update(float dt);

    // Accessors for particle data on host (after update, host data is synced for rendering)
    const ParticleData* getParticlesHost() const { return h_particles; }
    std::size_t getParticleCount() const { return particleCount; }

private:
    std::size_t particleCount;
    float boundaryRadius;
    float gravityX, gravityY;
    ParticleData* h_particles; // Host-side particle array (mirrors device data)
    ParticleData* d_particles; // Device-side particle array
    float* d_dx;               // Device array for X displacement offsets (collision resolution)
    float* d_dy;               // Device array for Y displacement offsets (collision resolution)
};
