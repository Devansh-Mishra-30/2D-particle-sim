// simulator_cuda.cu
#include <iostream>
#include <chrono>
#include "simulator_cuda.cuh"
#include <cmath>  // for sqrtf

// CUDA kernel: integrate motion (gravity, velocity, position) and handle boundary bounce for each particle
__global__ void integrateKernel(ParticleData* particles, std::size_t N, float dt,
                                float gravityX, float gravityY, float boundaryR) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    ParticleData& p = particles[i];
    // Apply gravity to velocity
    p.vx += gravityX * dt;
    p.vy += gravityY * dt;
    // Update position based on velocity
    p.px += p.vx * dt;
    p.py += p.vy * dt;

    // Enforce circular boundary (center at 0,0) with elastic bounce
    float dist2 = p.px * p.px + p.py * p.py;
    float maxRadius = boundaryR - p.radius;  // effective boundary radius for particle center
    if (dist2 > maxRadius * maxRadius) {
        float dist = sqrtf(dist2);
        if (dist != 0.0f) {
            // Normalized vector from center to particle position
            float nx = p.px / dist;
            float ny = p.py / dist;
            // Clamp position to boundary edge
            p.px = nx * maxRadius;
            p.py = ny * maxRadius;
            // Reflect velocity (if moving outward, invert that component)
            float vDotN = p.vx * nx + p.vy * ny;
            if (vDotN > 0.0f) {
                p.vx -= 2.0f * vDotN * nx;
                p.vy -= 2.0f * vDotN * ny;
            }
        }
    }
}

// CUDA kernel: detect overlaps and compute position offsets for particle-particle collisions
__global__ void collideKernel(const ParticleData* particles, float* d_dx, float* d_dy, std::size_t N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Each thread checks collisions of particle i with particles j > i
    float px_i = particles[i].px;
    float py_i = particles[i].py;
    float ri   = particles[i].radius;
    for (unsigned int j = i + 1; j < N; ++j) {
        float px_j = particles[j].px;
        float py_j = particles[j].py;
        float rj   = particles[j].radius;
        // Compute squared distance between particle i and j
        float dx = px_j - px_i;
        float dy = py_j - py_i;
        float dist2 = dx * dx + dy * dy;
        float minDist = ri + rj;
        if (dist2 < minDist * minDist) {
            float dist = sqrtf(dist2);
            if (dist == 0.0f) {
                // If particles are exactly overlapping, skip to avoid division by zero
                continue;
            }
            // Calculate overlap distance beyond allowed separation
            float overlap = 0.5f * (minDist - dist);
            // Normalized direction from particle i to particle j
            float nx = dx / dist;
            float ny = dy / dist;
            // Accumulate position adjustments for each particle (using atomics for thread safety)
            atomicAdd(&d_dx[i], -overlap * nx);
            atomicAdd(&d_dy[i], -overlap * ny);
            atomicAdd(&d_dx[j],  overlap * nx);
            atomicAdd(&d_dy[j],  overlap * ny);
        }
    }
}

// CUDA kernel: apply collision offsets to particle positions and enforce boundary constraints again
__global__ void applyOffsetsKernel(ParticleData* particles, float* d_dx, float* d_dy,
                                   std::size_t N, float boundaryR) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    ParticleData& p = particles[i];
    // Apply accumulated positional adjustments from collisions
    p.px += d_dx[i];
    p.py += d_dy[i];
    // (d_dx and d_dy are reset to 0 on the host between frames, so no need to clear here)
    // Check boundary again in case collision resolution pushed particle outside
    float dist2 = p.px * p.px + p.py * p.py;
    float maxRadius = boundaryR - p.radius;
    if (dist2 > maxRadius * maxRadius) {
        float dist = sqrtf(dist2);
        if (dist != 0.0f) {
            float nx = p.px / dist;
            float ny = p.py / dist;
            p.px = nx * maxRadius;
            p.py = ny * maxRadius;
            // Reflect velocity if particle was pushed outwards
            float vDotN = p.vx * nx + p.vy * ny;
            if (vDotN > 0.0f) {
                p.vx -= 2.0f * vDotN * nx;
                p.vy -= 2.0f * vDotN * ny;
            }
        }
    }
}

// Constructor: allocate device memory and copy initial particle data from host to device
Simulator::Simulator(ParticleData* hostArray, std::size_t count, float boundaryRadius,
                     float gravityX, float gravityY)
    : particleCount(count), boundaryRadius(boundaryRadius),
      gravityX(gravityX), gravityY(gravityY),
      h_particles(hostArray) {
    // Allocate device array for particles
    cudaMalloc(&d_particles, particleCount * sizeof(ParticleData));
    // Allocate device arrays for collision offsets
    cudaMalloc(&d_dx, particleCount * sizeof(float));
    cudaMalloc(&d_dy, particleCount * sizeof(float));
    // Copy initial particle data from host to device
    cudaMemcpy(d_particles, h_particles, particleCount * sizeof(ParticleData),
               cudaMemcpyHostToDevice);
}

// Destructor: free device memory
Simulator::~Simulator() {
    cudaFree(d_particles);
    cudaFree(d_dx);
    cudaFree(d_dy);
}

void Simulator::update(float dt) {
    // setup
    int threads = 256;
    int blocks  = (particleCount + threads - 1) / threads;

    // time points
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. integrate + boundary
    integrateKernel<<<blocks, threads>>>(d_particles, particleCount, dt,
                                         gravityX, gravityY, boundaryRadius);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    // 2. reset offsets
    cudaMemset(d_dx, 0, particleCount * sizeof(float));
    cudaMemset(d_dy, 0, particleCount * sizeof(float));

    // 3. collision detection
    collideKernel<<<blocks, threads>>>(d_particles, d_dx, d_dy, particleCount);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    // 4. apply offsets + boundary
    applyOffsetsKernel<<<blocks, threads>>>(d_particles, d_dx, d_dy,
                                            particleCount, boundaryRadius);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();

    // 5. copy back for rendering
    cudaMemcpy(h_particles, d_particles, particleCount * sizeof(ParticleData),
               cudaMemcpyDeviceToHost);
    auto t4 = std::chrono::high_resolution_clock::now();

    // compute durations
    std::chrono::duration<double,std::milli> dtIntegrate = t1 - t_start;
    std::chrono::duration<double,std::milli> dtCollide   = t2 - t1;
    std::chrono::duration<double,std::milli> dtApply     = t3 - t2;
    std::chrono::duration<double,std::milli> dtMemcpy    = t4 - t3;

    // print out
    static bool header_done = false;
    if (!header_done) {
        std::cout << "Integrate,Collide,ApplyOffsets,HostMemcpy\n";
        header_done = true;
    }
    std::cout
        << dtIntegrate.count() << ","
        << dtCollide.count()   << ","
        << dtApply.count()     << ","
        << dtMemcpy.count()    << "\n";
}

