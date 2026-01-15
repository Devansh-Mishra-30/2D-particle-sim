// particle_data.h
#pragma once
#include <cstdint>   // for uint8_t

// Structure representing a particle's data, optimized for GPU usage
struct ParticleData {
    float px, py;      // Position of the particle (x, y)
    float vx, vy;      // Velocity of the particle (x, y components)
    float radius;      // Radius of the particle
    uint8_t r, g, b, a; // Color components (RGBA)
};
