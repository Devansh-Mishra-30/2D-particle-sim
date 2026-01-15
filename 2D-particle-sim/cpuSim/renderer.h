#pragma once

#include <SFML/Graphics.hpp>

#include "simulator.h"

class Renderer
{
private:
    sf::RenderWindow& _window;
    sf::Texture _particle_texture;
    sf::VertexBuffer _particle_vbo;
    std::vector<sf::Vertex> _particle_vertices;

    void renderBackground(const Simulator& simulator) const;
    void renderObjects(const Simulator& simulator);

public:
    Renderer(sf::RenderWindow& window, int init_vbo_size = 0); // Ideally, set init_vbo_size to the maximum expected particles

    void render(const Simulator& simulator);
    void renderNaive(const Simulator& simulator) const;
};