#include <iostream>

#include "renderer.h"

Renderer::Renderer(sf::RenderWindow& window, int init_vbo_size /* =0 */) : _window(window), _particle_vbo(sf::PrimitiveType::Triangles, sf::VertexBuffer::Usage::Stream)
{
    if (!_particle_vbo.create(init_vbo_size)) 
    {
        std::cerr << "Vertex buffer initialization failed in Renderer::Renderer" << std::endl;
    }
    if (!_particle_texture.loadFromFile("assets/circle.png"))
    {
        std::cerr << "Texture initialization failed in Renderer::Renderer" << std::endl;
    }
}

void Renderer::render(const Simulator& simulator)
{
    renderBackground(simulator);
    renderObjects(simulator);
}

void Renderer::renderBackground(const Simulator& simulator) const
{
    // Render the boundary constraints:
    switch (simulator.getBoundaryType())
    {
        case BoundaryType::Circle:
        {
            const auto [boundary_center, boundary_radius] = simulator.getCircleBoundaryConstraints();
            sf::CircleShape constraint_background{boundary_radius};
            constraint_background.setOrigin({boundary_radius, boundary_radius}); // Set origin of circle's coordinate system to its center
            constraint_background.setFillColor(sf::Color::Black);
            constraint_background.setPosition({boundary_center.x, boundary_center.y});
            constraint_background.setPointCount(128);
            _window.draw(constraint_background);
            break;
        }
        case BoundaryType::Rectangle:
        {
            const auto [boundary_center, boundary_size] = simulator.getRectangleBoundaryConstraints();
            sf::RectangleShape constraint_background{boundary_size};
            constraint_background.setOrigin({0.5f * boundary_size.x, 0.5f * boundary_size.y}); // Set origin of rectangle's coordinate system to its center
            constraint_background.setFillColor(sf::Color::Black);
            constraint_background.setPosition({boundary_center.x, boundary_center.y});
            _window.draw(constraint_background);
            break;
        }
        default:
        {
            std::cerr << "Unsupported BoundaryType used in Renderer::render" << std::endl;
            break;
        }
    }
}

void Renderer::renderObjects(const Simulator& simulator)
{
    // For each particle, render a square using two triangles. Use a circle texture on the square to display the circle.
    // (This drastically reduces the number of vertices used as opposed to sf::CircleShape or sf::Primitives::TriangleFan)
    // Storing vertices for all particles within a single buffer reduces the # of draw calls to 1.
    const auto& particles = simulator.getParticles();
    _particle_vertices.resize(particles.size() * 6); // allocate space for 2 triangles per particle

    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p             = particles[i];
        const sf::Vector2f center = p.getPosition();
        const float radius        = p.getRadius();
        const sf::Color color     = p.getColor();
        
        // Corners of the square (top-left, top-right, bottom-right, bottom-left)
        const sf::Vector2f offsets[4] = {
            {-radius, -radius}, {radius, -radius},
            {radius, radius}, {-radius, radius}
        };
        
        // Texture coordinates (top-left, top-right, bottom-right, bottom-left)
        const sf::Vector2f texSize(_particle_texture.getSize());
        const sf::Vector2f texCoords[4] = {
            {0.f, 0.f}, {texSize.x, 0.f},
            {texSize.x, texSize.y}, {0.f, texSize.y}
        };
        
        const int objectIdx = i * 6;

        // First triangle (top-left, top-right, bottom-right)
        _particle_vertices[objectIdx].position  = center + offsets[0];
        _particle_vertices[objectIdx].texCoords = texCoords[0];
        _particle_vertices[objectIdx].color     = color;
        
        _particle_vertices[objectIdx + 1].position  = center + offsets[1];
        _particle_vertices[objectIdx + 1].texCoords = texCoords[1];
        _particle_vertices[objectIdx + 1].color     = color;
        
        _particle_vertices[objectIdx + 2].position  = center + offsets[2];
        _particle_vertices[objectIdx + 2].texCoords = texCoords[2];
        _particle_vertices[objectIdx + 2].color     = color;
        
        // Second triangle (bottom-right, bottom-left, top-left)
        _particle_vertices[objectIdx + 3].position  = center + offsets[2];
        _particle_vertices[objectIdx + 3].texCoords = texCoords[2];
        _particle_vertices[objectIdx + 3].color     = color;
        
        _particle_vertices[objectIdx + 4].position  = center + offsets[3];
        _particle_vertices[objectIdx + 4].texCoords = texCoords[3];
        _particle_vertices[objectIdx + 4].color     = color;
        
        _particle_vertices[objectIdx + 5].position  = center + offsets[0];
        _particle_vertices[objectIdx + 5].texCoords = texCoords[0];
        _particle_vertices[objectIdx + 5].color     = color;
    }

    sf::RenderStates states;
    states.texture = &_particle_texture;
    if (!_particle_vbo.update(_particle_vertices.data(), _particle_vertices.size(), 0))
    {
        std::cerr << "Vertex buffer update failed in Renderer::renderObjects" << std::endl;
    }
    _window.draw(_particle_vbo, states);
}


void Renderer::renderNaive(const Simulator& simulator) const
{
    renderBackground(simulator);

    // Render particles:
    // To avoid constructing new shapes each iteration, one circle is declared and modified by each particle
    sf::CircleShape circle{1.0f};
    circle.setPointCount(32);
    circle.setOrigin({1.0f, 1.0f});
    const auto& particles = simulator.getParticles();
    for (const auto& p : particles) {
        const auto& position = p.getPosition();
        float radius = p.getRadius();
        sf::Color color = p.getColor();

        circle.setPosition(position);
        circle.setScale({radius, radius});
        circle.setFillColor(color);
        _window.draw(circle);
    }
}