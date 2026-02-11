#include "simulator_cuda.cuh"
#include <cmath>
#include <iostream>

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " | " << cudaGetErrorString(err) << "\n";
        std::abort();
    }
}

__global__ void integrateKernel(ParticleData* particles,
                                std::size_t N,
                                float dt,
                                float gravityX,
                                float gravityY,
                                float boundaryR) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    ParticleData& p = particles[i];

    p.vx += gravityX * dt;
    p.vy += gravityY * dt;

    p.px += p.vx * dt;
    p.py += p.vy * dt;

    float dist2 = p.px * p.px + p.py * p.py;
    float maxRadius = boundaryR - p.radius;
    float maxR2 = maxRadius * maxRadius;

    if (dist2 > maxR2) {
        float dist = sqrtf(dist2);
        if (dist > 0.0f) {
            float nx = p.px / dist;
            float ny = p.py / dist;

            p.px = nx * maxRadius;
            p.py = ny * maxRadius;

            float vDotN = p.vx * nx + p.vy * ny;
            if (vDotN > 0.0f) {
                p.vx -= 2.0f * vDotN * nx;
                p.vy -= 2.0f * vDotN * ny;
            }
        }
    }
}

__global__ void collideKernel(const ParticleData* particles,
                              float* d_dx,
                              float* d_dy,
                              std::size_t N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px_i = particles[i].px;
    float py_i = particles[i].py;
    float ri   = particles[i].radius;

    for (unsigned int j = i + 1; j < N; ++j) {
        float px_j = particles[j].px;
        float py_j = particles[j].py;
        float rj   = particles[j].radius;

        float dx = px_j - px_i;
        float dy = py_j - py_i;
        float dist2 = dx * dx + dy * dy;

        float minDist = ri + rj;
        float minDist2 = minDist * minDist;

        if (dist2 < minDist2) {
            float dist = sqrtf(dist2);
            if (dist == 0.0f) continue;

            float overlap = 0.5f * (minDist - dist);
            float nx = dx / dist;
            float ny = dy / dist;

            atomicAdd(&d_dx[i], -overlap * nx);
            atomicAdd(&d_dy[i], -overlap * ny);
            atomicAdd(&d_dx[j],  overlap * nx);
            atomicAdd(&d_dy[j],  overlap * ny);
        }
    }
}

__global__ void applyOffsetsKernel(ParticleData* particles,
                                   const float* d_dx,
                                   const float* d_dy,
                                   std::size_t N,
                                   float boundaryR) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    ParticleData& p = particles[i];

    p.px += d_dx[i];
    p.py += d_dy[i];

    float dist2 = p.px * p.px + p.py * p.py;
    float maxRadius = boundaryR - p.radius;
    float maxR2 = maxRadius * maxRadius;

    if (dist2 > maxR2) {
        float dist = sqrtf(dist2);
        if (dist > 0.0f) {
            float nx = p.px / dist;
            float ny = p.py / dist;

            p.px = nx * maxRadius;
            p.py = ny * maxRadius;

            float vDotN = p.vx * nx + p.vy * ny;
            if (vDotN > 0.0f) {
                p.vx -= 2.0f * vDotN * nx;
                p.vy -= 2.0f * vDotN * ny;
            }
        }
    }
}

Simulator::Simulator(ParticleData* hostArray,
                     std::size_t count,
                     float boundaryRadiusIn,
                     float gravityXIn,
                     float gravityYIn)
    : particleCount(count),
      boundaryRadius(boundaryRadiusIn),
      gravityX(gravityXIn),
      gravityY(gravityYIn),
      h_particles(hostArray),
      d_particles(nullptr),
      d_dx(nullptr),
      d_dy(nullptr),
      stream(nullptr),
      evStart(nullptr),
      evStop(nullptr) {

    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");

    cudaCheck(cudaMalloc(&d_particles, particleCount * sizeof(ParticleData)), "cudaMalloc d_particles");
    cudaCheck(cudaMalloc(&d_dx, particleCount * sizeof(float)), "cudaMalloc d_dx");
    cudaCheck(cudaMalloc(&d_dy, particleCount * sizeof(float)), "cudaMalloc d_dy");

    cudaCheck(cudaMemcpyAsync(d_particles,
                              h_particles,
                              particleCount * sizeof(ParticleData),
                              cudaMemcpyHostToDevice,
                              stream),
              "cudaMemcpyAsync H2D particles");

    cudaCheck(cudaMemsetAsync(d_dx, 0, particleCount * sizeof(float), stream), "cudaMemsetAsync d_dx");
    cudaCheck(cudaMemsetAsync(d_dy, 0, particleCount * sizeof(float), stream), "cudaMemsetAsync d_dy");

    cudaCheck(cudaEventCreate(&evStart), "cudaEventCreate start");
    cudaCheck(cudaEventCreate(&evStop), "cudaEventCreate stop");

    cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize init");
}

Simulator::~Simulator() {
    if (evStart) cudaEventDestroy(evStart);
    if (evStop) cudaEventDestroy(evStop);
    if (d_particles) cudaFree(d_particles);
    if (d_dx) cudaFree(d_dx);
    if (d_dy) cudaFree(d_dy);
    if (stream) cudaStreamDestroy(stream);
}

void Simulator::runKernels(float dt) {
    int threads = 256;
    int blocks = static_cast<int>((particleCount + threads - 1) / threads);

    integrateKernel<<<blocks, threads, 0, stream>>>(d_particles, particleCount, dt,
                                                   gravityX, gravityY, boundaryRadius);
    cudaCheck(cudaGetLastError(), "integrateKernel launch");

    cudaCheck(cudaMemsetAsync(d_dx, 0, particleCount * sizeof(float), stream), "cudaMemsetAsync d_dx");
    cudaCheck(cudaMemsetAsync(d_dy, 0, particleCount * sizeof(float), stream), "cudaMemsetAsync d_dy");

    collideKernel<<<blocks, threads, 0, stream>>>(d_particles, d_dx, d_dy, particleCount);
    cudaCheck(cudaGetLastError(), "collideKernel launch");

    applyOffsetsKernel<<<blocks, threads, 0, stream>>>(d_particles, d_dx, d_dy, particleCount, boundaryRadius);
    cudaCheck(cudaGetLastError(), "applyOffsetsKernel launch");
}

void Simulator::copyToHost() {
    cudaCheck(cudaMemcpyAsync(h_particles,
                              d_particles,
                              particleCount * sizeof(ParticleData),
                              cudaMemcpyDeviceToHost,
                              stream),
              "cudaMemcpyAsync D2H particles");
    cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize copyToHost");
}

void Simulator::update(float dt, bool syncToHost) {
    runKernels(dt);
    if (syncToHost) {
        copyToHost();
        return;
    }
}

float Simulator::benchmarkSteps(float dt, int warmupSteps, int measuredSteps) {
    if (warmupSteps < 0) warmupSteps = 0;
    if (measuredSteps <= 0) measuredSteps = 1;

    for (int i = 0; i < warmupSteps; ++i) {
        runKernels(dt);
    }
    cudaCheck(cudaStreamSynchronize(stream), "warmup sync");

    cudaCheck(cudaEventRecord(evStart, stream), "event record start");
    for (int i = 0; i < measuredSteps; ++i) {
        runKernels(dt);
    }
    cudaCheck(cudaEventRecord(evStop, stream), "event record stop");
    cudaCheck(cudaEventSynchronize(evStop), "event sync stop");

    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, evStart, evStop), "event elapsed time");
    return ms / static_cast<float>(measuredSteps);
}
