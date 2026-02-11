#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include "particle_data.h"

class Renderer {
public:
    explicit Renderer(std::size_t maxParticles);
    void drawParticles(sf::RenderWindow& window, const ParticleData* particles, std::size_t count);

private:
    sf::Texture circleTexture;
    sf::VertexBuffer vertexBuffer;
    std::vector<sf::Vertex> vertices;
    unsigned int textureWidth = 0;
    unsigned int textureHeight = 0;
};
