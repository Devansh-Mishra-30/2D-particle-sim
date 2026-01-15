#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <optional>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "particle_data.h"
#include "simulator_cuda.cuh"
#include "renderer.h"

int main() {
    const unsigned int windowWidth = 800;
    const unsigned int windowHeight = 800;
    const float boundaryRadius = 350.0f;
    const std::size_t particleCount = 1000;
    const sf::Vector2f gravity(0.0f, 100.0f);

    // Create SFML window
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(windowWidth, windowHeight)), "GPU Particle Simulation");

    // Center the coordinate system so that (0, 0) is at the center of the screen
    sf::View customView(sf::Vector2f(0.f, 0.f), sf::Vector2f(windowWidth, windowHeight));
    window.setView(customView);
    window.setFramerateLimit(60);

    // Initialize particles
    std::vector<ParticleData> particles(particleCount);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    float maxInitialDist = boundaryRadius - 20.0f;
    for (std::size_t i = 0; i < particleCount; ++i) {
        float radius = 2.0f + static_cast<float>(std::rand() % 11);
        float angle = (static_cast<float>(std::rand()) / RAND_MAX) * 2 * 3.14159265f;
        float randUnit = static_cast<float>(std::rand()) / RAND_MAX;
        float rDist = maxInitialDist * std::sqrt(randUnit);
        particles[i].px = rDist * std::cos(angle);
        particles[i].py = rDist * std::sin(angle);
        particles[i].vx = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 100.0f;
        particles[i].vy = 0.0f;
        particles[i].radius = radius;
        particles[i].r = std::rand() % 256;
        particles[i].g = std::rand() % 256;
        particles[i].b = std::rand() % 256;
        particles[i].a = 255;
    }

    // Simulator + Renderer
    Simulator simulator(particles.data(), particleCount, boundaryRadius, gravity.x, gravity.y);
    Renderer renderer(particleCount);

    sf::Clock clock;
    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (auto key = event->getIf<sf::Event::KeyPressed>()) {
                if (key->scancode == sf::Keyboard::Scancode::Escape)
                    window.close();
            }
        }

        float dt = clock.restart().asSeconds();
        simulator.update(dt);

        window.clear(sf::Color::Black);
        renderer.drawParticles(window, simulator.getParticlesHost(), simulator.getParticleCount());
        window.display();
    }

    return 0;
}
