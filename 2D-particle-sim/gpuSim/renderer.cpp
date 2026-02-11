#include "renderer.h"
#include <stdexcept>
#include <iostream>

Renderer::Renderer(std::size_t maxParticles) {
    if (!circleTexture.loadFromFile("assets/circle.png")) {
        throw std::runtime_error("Failed to load texture: assets/circle.png");
    }
    circleTexture.setSmooth(true);

    if (!vertexBuffer.create(maxParticles * 6)) {
        std::cerr << "Failed to create vertex buffer!\n";
    }

    vertexBuffer.setPrimitiveType(sf::PrimitiveType::Triangles);
    vertexBuffer.setUsage(sf::VertexBuffer::Usage::Stream);

    const sf::Vector2u texSize = circleTexture.getSize();
    textureWidth = texSize.x;
    textureHeight = texSize.y;

    vertices.resize(maxParticles * 6);
}

void Renderer::drawParticles(sf::RenderWindow& window, const ParticleData* particles, std::size_t count) {
    std::size_t vIndex = 0;

    for (std::size_t i = 0; i < count; ++i) {
        const ParticleData& p = particles[i];
        const float px = p.px;
        const float py = p.py;
        const float r  = p.radius;

        const float left   = px - r;
        const float right  = px + r;
        const float top    = py - r;
        const float bottom = py + r;

        const sf::Color color(p.r, p.g, p.b, p.a);

        vertices[vIndex].position  = {left, top};
        vertices[vIndex].texCoords = {0.f, 0.f};
        vertices[vIndex].color     = color;
        ++vIndex;

        vertices[vIndex].position  = {right, top};
        vertices[vIndex].texCoords = {static_cast<float>(textureWidth), 0.f};
        vertices[vIndex].color     = color;
        ++vIndex;

        vertices[vIndex].position  = {right, bottom};
        vertices[vIndex].texCoords = {static_cast<float>(textureWidth), static_cast<float>(textureHeight)};
        vertices[vIndex].color     = color;
        ++vIndex;

        vertices[vIndex].position  = {right, bottom};
        vertices[vIndex].texCoords = {static_cast<float>(textureWidth), static_cast<float>(textureHeight)};
        vertices[vIndex].color     = color;
        ++vIndex;

        vertices[vIndex].position  = {left, bottom};
        vertices[vIndex].texCoords = {0.f, static_cast<float>(textureHeight)};
        vertices[vIndex].color     = color;
        ++vIndex;

        vertices[vIndex].position  = {left, top};
        vertices[vIndex].texCoords = {0.f, 0.f};
        vertices[vIndex].color     = color;
        ++vIndex;
    }

    const bool ok = vertexBuffer.update(vertices.data(), count * 6, 0);
    if (!ok) {
        std::cerr << "VertexBuffer update failed\n";
        return;
    }

    sf::RenderStates states;
    states.texture = &circleTexture;
    window.draw(vertexBuffer, states);
}
