#include "simulator.h"

#include <algorithm>
#include <cmath>

const float Particle::DEFAULT_RADIUS = 5.0f;
const sf::Vector2f Particle::DEFAULT_ACCELERATION(0.0f, 0.0f);
const sf::Vector2f Simulator::DEFAULT_BOUNDARY_CENTER(0.0f, 0.0f);
const float Simulator::DEFAULT_BOUNDARY_RADIUS = 200.0f;
const sf::Vector2f Simulator::DEFAULT_GRAVITY(0.0f, 980.0f);
const float Simulator::DEFAULT_COLLISION_COEF = 0.75f;

Particle& Simulator::addParticle(const Particle& particle) {
    auto& p = _particles.emplace_back(particle);
    p.accelerate(_gravity);
    return p;
}

void Simulator::updatePositions() {
    #pragma omp parallel for
    for (int i = 0; i < _particles.size(); i++) {
        _particles[i].evolve(_dt);
    }
}

void Simulator::resolveCollisions() {
    // particle-particle collision
    int particle_count = _particles.size();

    #pragma omp parallel for
    for (int i = 0; i < particle_count; ++i) {
        auto& p1 = _particles[i];
        float mass_1 = p1._radius * p1._radius;
        for (int j = i + 1; j < particle_count; ++j) {
            auto& p2 = _particles[j];
            sf::Vector2f dist = p1._cur_position - p2._cur_position;
            float min_dist = p1._radius + p2._radius;
            if (dist.lengthSquared() < min_dist * min_dist) {
                float mass_2 = p2._radius * p2._radius;
                float dist_len = dist.length();
                const sf::Vector2f n = dist / dist_len;
                const float delta = 0.5f * _collision_coef * (dist_len - min_dist);

                // Update positions
                p1._cur_position -= n * delta * (mass_2 / (mass_1 + mass_2));
                p2._cur_position += n * delta * (mass_1 / (mass_1 + mass_2));
            }
        }
    }
}

void Simulator::resolveCollisionsSweepPrune() {
    // Sort particles in increasing order of left side x coordinate
    std::sort(_particles.begin(), _particles.end(), [](const Particle& p1, const Particle& p2) {
        return p1._cur_position.x - p1._radius < p2._cur_position.x - p2._radius;
    });

    int particle_count = _particles.size();

    #pragma omp parallel for
    for (int i = 0; i < particle_count; ++i) {
        auto& p1 = _particles[i];
        float mass_1 = p1._radius * p1._radius;
        for (int j = i + 1; j < particle_count; ++j) {
            auto& p2 = _particles[j];

            // If the left edge of the second particle is > the right edge of the first particle,
            // it is not possible for this particle (or particles after) to collide
            if (p2._cur_position.x - p2._radius > p1._cur_position.x + p1._radius) break;

            sf::Vector2f dist = p1._cur_position - p2._cur_position;
            float min_dist = p1._radius + p2._radius;
            if (dist.lengthSquared() < min_dist * min_dist) {
                float mass_2 = p2._radius * p2._radius;
                float dist_len = dist.length();
                const sf::Vector2f n = dist / dist_len;
                const float delta = 0.5f * _collision_coef * (dist_len - min_dist);

                // Update positions
                p1._cur_position -= n * delta * (mass_2 / (mass_1 + mass_2));
                p2._cur_position += n * delta * (mass_1 / (mass_1 + mass_2));
            }
        }
    }
}

void Simulator::applyBoundaryConstraints() {
    // boundary collision
    switch (_boundary_type)
    {
        case BoundaryType::Circle:
        {
            #pragma omp parallel for
            for (int i = 0; i < _particles.size(); ++i) {
                auto& particle = _particles[i];
                const sf::Vector2f dist_v = _boundary_center - particle._cur_position;
                const float dist = dist_v.length();
                if (dist > (_boundary_radius - particle._radius)) {
                    const sf::Vector2f n = dist_v / dist;
                    particle._cur_position = _boundary_center - n * (_boundary_radius - particle._radius);
                }
            }
            break;
        }
        case BoundaryType::Rectangle:
        {
            // const auto& [left, right, top, bottom] = getBoundaryBorders(); // was giving errors with openmp?
            float left, right, top, bottom;
            auto borders = getBoundaryBorders();
            left   = borders[0];
            right  = borders[1];
            top    = borders[2];
            bottom = borders[3];

            // If a particle hits the border, clip it to the border and negate its velocity
            #pragma omp parallel for
            for (int i = 0; i < _particles.size(); i++) {
                auto& particle = _particles[i];
                const float margin = particle._radius;
        
                if (particle._cur_position.x > right - margin) {
                    particle._cur_position.x = right - margin;
                    particle._last_position.x = 2.0f * particle._cur_position.x - particle._last_position.x;
                } 
                else if (particle._cur_position.x < left + margin) {
                    particle._cur_position.x = left + margin;
                    particle._last_position.x = 2.0f * particle._cur_position.x - particle._last_position.x;
                }
                if (particle._cur_position.y > bottom - margin) {
                    particle._cur_position.y = bottom - margin;
                    particle._last_position.y = 2.0f * particle._cur_position.y - particle._last_position.y;
                } 
                else if (particle._cur_position.y < top + margin) {
                    particle._cur_position.y = top + margin;
                    particle._last_position.y = 2.0f * particle._cur_position.y - particle._last_position.y;
                }
            }
            break;
        }
        default:
        {
            std::cerr << "Invalid BoundaryType used in Simulator::applyBoundaryConstraints" << std::endl;
            break;
        }
    }
}

void Simulator::evolve() {
    for (int i = 0; i < _substeps; ++i) {
        resolveCollisionsSweepPrune();
        applyBoundaryConstraints();
        updatePositions();
    }
}
