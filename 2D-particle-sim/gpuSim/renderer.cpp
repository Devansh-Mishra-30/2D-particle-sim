// renderer.cpp
#include "renderer.h"
#include <stdexcept>
#include <iostream>

Renderer::Renderer(std::size_t maxParticles) {
    if (!circleTexture.loadFromFile("assets/circle.png")) {
        throw std::runtime_error("Failed to load texture: assets/circle.png");
    }
    circleTexture.setSmooth(true);

    if (!vertexBuffer.create(maxParticles * 6)) {
        std::cerr << "Failed to create vertex buffer!" << std::endl;
    }

    vertexBuffer.setPrimitiveType(sf::PrimitiveType::Triangles);
    vertexBuffer.setUsage(sf::VertexBuffer::Usage::Stream);

    sf::Vector2u texSize = circleTexture.getSize();
    textureWidth = texSize.x;
    textureHeight = texSize.y;

    vertices.resize(maxParticles * 6); // allocate once
}

void Renderer::drawParticles(sf::RenderWindow& window, const ParticleData* particles, std::size_t count) {
    std::size_t vIndex = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const ParticleData& p = particles[i];
        float px = p.px;
        float py = p.py;
        float r  = p.radius;

        float left   = px - r;
        float right  = px + r;
        float top    = py - r;
        float bottom = py + r;

        sf::Color color(p.r, p.g, p.b, p.a);

        // Triangle 1
        vertices[vIndex].position   = {left,  top};
        vertices[vIndex].texCoords  = {0.f, 0.f};
        vertices[vIndex].color      = color;
        vIndex++;

        vertices[vIndex].position   = {right, top};
        vertices[vIndex].texCoords  = {static_cast<float>(textureWidth), 0.f};
        vertices[vIndex].color      = color;
        vIndex++;

        vertices[vIndex].position   = {right, bottom};
        vertices[vIndex].texCoords  = {static_cast<float>(textureWidth), static_cast<float>(textureHeight)};
        vertices[vIndex].color      = color;
        vIndex++;

        // Triangle 2
        vertices[vIndex].position   = {right, bottom};
        vertices[vIndex].texCoords  = {static_cast<float>(textureWidth), static_cast<float>(textureHeight)};
        vertices[vIndex].color      = color;
        vIndex++;

        vertices[vIndex].position   = {left,  bottom};
        vertices[vIndex].texCoords  = {0.f, static_cast<float>(textureHeight)};
        vertices[vIndex].color      = color;
        vIndex++;

        vertices[vIndex].position   = {left,  top};
        vertices[vIndex].texCoords  = {0.f, 0.f};
        vertices[vIndex].color      = color;
        vIndex++;
    }

    vertexBuffer.update(vertices.data(), count * 6, 0);
    sf::RenderStates states;
    states.texture = &circleTexture;
    window.draw(vertexBuffer, states);
}
