// renderer.h
#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include "particle_data.h"

class Renderer {
public:
    Renderer(std::size_t maxParticles);
    // Update vertex buffer and draw all particles to the window
    void drawParticles(sf::RenderWindow& window, const ParticleData* particles, std::size_t count);

private:
    sf::Texture circleTexture;       // Texture used for a particle (circle image)
    sf::VertexBuffer vertexBuffer;   // Vertex buffer for all particle vertices
    std::vector<sf::Vertex> vertices; // CPU-side storage for vertices to update the VBO
    unsigned int textureWidth;
    unsigned int textureHeight;
};
