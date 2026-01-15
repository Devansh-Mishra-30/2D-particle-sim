#include <SFML/Graphics.hpp>

#include <iostream>
#include <random>

#include "renderer.h"
#include "simulator.h"

int main(int argc, char* argv[]) {
    constexpr int window_height = 950;
    constexpr int window_width = 950;

    const sf::Vector2f boundary_center({static_cast<float>(window_width) * 0.5f, static_cast<float>(window_height) * 0.5f});
    const float boundary_radius = 400.0f;
    const float particle_radius_max = 10.0f;
    const float particle_radius_min = 2.0f;
    const float init_velocity_max = 500.0f;
    const float init_velocity_min = -500.0f;

    const sf::Vector2f born_position(boundary_center);
    const sf::Vector2f gravity(0.0, -980.0);
    const int frame_rate = 60;
    const float DT = 1.0f / frame_rate;

    // Initialize SFML window
    sf::ContextSettings settings;
    settings.antiAliasingLevel = 1;
    sf::RenderWindow window(sf::VideoMode({window_width, window_height}), "Particle Simulation", sf::Style::Default, sf::State::Windowed, settings);
    window.setFramerateLimit(frame_rate);

    // Create view to invert y-axis (since (0, 0) is the top left corner instead of bottom left)
    sf::View view = window.getDefaultView();
    view.setSize({static_cast<float>(window_width), static_cast<float>(-window_height)}); 

    int particle_count = 20000;
    if (argc > 1) {
        particle_count = std::atoi(argv[1]);
    }

    try {
        // init renderer
        Renderer renderer{window, particle_count};

        // init simulator
        Simulator simulator(DT, particle_count, boundary_radius, boundary_center, gravity);                            // Circular boundary
        // Simulator simulator(DT, particle_count, {window_width, window_height}, boundary_center, gravity);              // Rectangle boundary
        // Simulator simulator(DT, particle_count, {0.7f * window_width, 0.7f * window_width}, boundary_center, gravity); // Rectangle boundary (smaller than window)
        simulator.setSubsteps(8);

        // init fps / object counter
        sf::Clock fps_clock;
        sf::Clock update_clock;
        sf::Font font("assets/Arial.ttf");
        sf::Text fpsText(font);
        fpsText.setCharacterSize(20);
        fpsText.setFillColor(sf::Color::Black);
        fpsText.setOutlineColor(sf::Color::White);
        fpsText.setOutlineThickness(1.0f);
        fpsText.setPosition({10, 10});
        sf::Text objectCountText(font);
        objectCountText.setCharacterSize(20);
        objectCountText.setFillColor(sf::Color::Black);
        objectCountText.setOutlineColor(sf::Color::White);
        objectCountText.setOutlineThickness(1.0f);
        objectCountText.setPosition({10, 30});

        // random generator
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> vel_dist(init_velocity_min, init_velocity_max);
        std::uniform_real_distribution<float> radius_dist(particle_radius_min, particle_radius_max);
        std::uniform_int_distribution<uint8_t> color_dist(0, 255);

        while (window.isOpen())
        {
            // Calculate FPS
            if (update_clock.getElapsedTime().asSeconds() >= 0.1f)
            {
                float currentTime = fps_clock.getElapsedTime().asSeconds();
                float fps = 1.0f / currentTime;
                fpsText.setString("FPS: " + std::to_string(static_cast<int>(fps)));
                update_clock.restart();
            }
            fps_clock.restart();

            // Get current number of particles
            objectCountText.setString("# Particles: " + std::to_string(simulator.getNumParticles()));
    
            // Exit if the user closes the window or hits the escape button
            while (const std::optional event = window.pollEvent())
            {
                if (event->is<sf::Event::Closed>())
                { 
                    window.close();
                } 
                else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>())
                {
                    if (keyPressed->scancode == sf::Keyboard::Scancode::Escape)
                        window.close();
                }
            }

            // add new particles
            if (simulator.getNumParticles() < particle_count) {
                Particle p;
                p.setRadius(radius_dist(gen));
                p.setColor({color_dist(gen), color_dist(gen), color_dist(gen)});

                sf::Vector2f last_position = born_position - sf::Vector2f(vel_dist(gen), vel_dist(gen)) * DT;
                // sf::Vector2f last_position = born_position - sf::Vector2f(200, -50) * DT; // Debug spawn
                p.setPosition(last_position, born_position);

                simulator.addParticle(p);
            }

            // evolve particles
            simulator.evolve();

            // render the simulation
            window.clear(sf::Color::White);
            window.setView(view);                    // Invert y-axis for objects
            renderer.render(simulator);
            window.setView(window.getDefaultView()); // Reset to default view before rendering text
            window.draw(fpsText);
            window.draw(objectCountText);
            window.display();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}