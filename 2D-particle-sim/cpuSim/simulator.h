#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/System/Vector3.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

enum class BoundaryType {Circle, Rectangle};

class Particle {
    static const float DEFAULT_RADIUS;
    static const sf::Vector2f DEFAULT_ACCELERATION;

    sf::Vector2f _last_position;
    sf::Vector2f _cur_position;
    sf::Vector2f _accelaration;
    float _radius;
    sf::Color _color;

    friend class Simulator;

   public:
    Particle() : _last_position(0.0f, 0.0f), _cur_position(0.0f, 0.0f), _accelaration(DEFAULT_ACCELERATION), _radius(DEFAULT_RADIUS), _color(sf::Color::White) {}

    Particle(const sf::Vector2f& last_position, const sf::Vector2f& cur_position, const float& radius = DEFAULT_RADIUS)
        : _last_position(last_position), _cur_position(cur_position), _accelaration(DEFAULT_ACCELERATION), _radius(radius), _color(sf::Color::White) {}

    sf::Vector2f accelerate(const sf::Vector2f& acceleration) {
        _accelaration += acceleration;
        return _accelaration;
    }

    void setAcceleration(const sf::Vector2f& acceleration) {
        _accelaration = acceleration;
    }

    void setPosition(const sf::Vector2f& last_position, const sf::Vector2f& cur_position) {
        _last_position = last_position;
        _cur_position = cur_position;
    }

    void setRadius(const float& radius) {
        _radius = radius;
    }

    const sf::Vector2f& getPosition() const {
        return _cur_position;
    }

    float getRadius() const {
        return _radius;
    }

    sf::Color getColor() const {
        return _color;
    }

    void setColor(sf::Color color) {
        _color = color;
    }

    void evolve(const float& dt) {
        sf::Vector2f next = _cur_position * 2.0f - _last_position + _accelaration * dt * dt;
        _last_position = _cur_position;
        _cur_position = next;
    }
};

class Simulator {
    static const sf::Vector2f DEFAULT_BOUNDARY_CENTER;
    static const float DEFAULT_BOUNDARY_RADIUS;
    static const sf::Vector2f DEFAULT_GRAVITY;
    static const float DEFAULT_COLLISION_COEF;

    int _max_particles;              // Maximum # particles the simulation will spawn
    std::vector<Particle> _particles;

    BoundaryType _boundary_type;
    float _boundary_radius;          // Radius for circlular boundary constraint
    sf::Vector2f _boundary_size;     // Width, Height for rectangular boundary constraints
    sf::Vector2f _boundary_center;
    sf::Vector2f _gravity;
    
    float _dt;
    float _substeps;

    float _collision_coef;

    /**
     * \brief handle particle-boundary collisions
     */
    void applyBoundaryConstraints();

    /**
     * \brief handle particle-particle collisions 
     */
    void resolveCollisions();

    /**
     * \brief handle particle-particle collisions using sweep and prune optimization
     */
    void resolveCollisionsSweepPrune();

    /**
     * \brief update all particles' position
     */
    void updatePositions();

public:
    Simulator(const float& dt, int max_particles, const float& boundary_radius = DEFAULT_BOUNDARY_RADIUS, const sf::Vector2f& boundary_center = DEFAULT_BOUNDARY_CENTER, const sf::Vector2f& gravity = DEFAULT_GRAVITY)
        : _max_particles(max_particles), _boundary_type(BoundaryType::Circle), _boundary_radius(boundary_radius), _boundary_center(boundary_center), _gravity(gravity), _dt(dt), _substeps(1), _collision_coef(DEFAULT_COLLISION_COEF) 
    {
        _particles.reserve(_max_particles);
    }

    Simulator(const float& dt, int max_particles, const sf::Vector2f& bbox_size, const sf::Vector2f& boundary_center = DEFAULT_BOUNDARY_CENTER, const sf::Vector2f& gravity = DEFAULT_GRAVITY)
        : _max_particles(max_particles), _boundary_type(BoundaryType::Rectangle), _boundary_size(bbox_size), _boundary_center(boundary_center), _gravity(gravity), _dt(dt), _substeps(1), _collision_coef(DEFAULT_COLLISION_COEF) 
    {
        _particles.reserve(_max_particles);
    }

    /**
     * \brief set the collision coefficient between particles
     */
    void setCollisionCoef(const float& coef) {
        _collision_coef = coef;
    }

    /**
     * \brief add new particle to the simulator
     * \param new paticle
     */
    Particle& addParticle(const Particle& particle);

    /**
     * \brief calculate new position of particles in dt time
     */
    void evolve();

    /**
     * \brief get substeps
    */
    int getSubsteps() const {
        return _substeps;
    }

    /**
     * \brief get particles
    */
    const std::vector<Particle>& getParticles() const {
        return _particles;
    }

    /**
     * \brief get number of particles
    */
    int getNumParticles() const {
        return _particles.size();
    }

    /**
     * \brief get max number of particles the simulation will spawn
    */
    int getMaxParticles() const {
        return _max_particles;
    }

    /**
     * \brief get type of boundary
    */
    const BoundaryType& getBoundaryType() const {
        return _boundary_type;
    }

    /**
     * \brief get constraints of the boundary circle: center coordinates, radius
    */
    std::pair<sf::Vector2f, float> getCircleBoundaryConstraints() const {
        return {_boundary_center, _boundary_radius};
    }

    /**
     * \brief get constraints of the boundary rectangle: center coordinates (x, y), size (width, height)
    */
    std::pair<sf::Vector2f, sf::Vector2f> getRectangleBoundaryConstraints() const {
        return {_boundary_center, _boundary_size};
    }

    /**
     * \brief get the bounding box [left, right, top, bottom] of the border constraints
    */
    std::array<float, 4> getBoundaryBorders() const {
        float half_width  = 0.0f;
        float half_height = 0.0f;
        switch (_boundary_type)
        {
            case BoundaryType::Circle:
            {
                half_width = half_height = _boundary_radius;
                break;
            }
            case BoundaryType::Rectangle:
            {
                half_width  = 0.5f * _boundary_size.x;
                half_height = 0.5f * _boundary_size.y;
                break;
            }
            default:
            {
                std::cerr << "Unsupported BoundaryType used in Simulator::getBoundaryBorders" << std::endl;
                break;
            }
        }
        const float left   = _boundary_center.x - half_width;
        const float right  = _boundary_center.x + half_width;
        const float top    = _boundary_center.y - half_height;
        const float bottom = _boundary_center.y + half_height;
        return {left, right, top, bottom};
    }

    /**
     * \brief set substeps
     */
    void setSubsteps(const int& substeps) {
        _substeps = substeps;
        _dt = _dt / substeps;
    }

    /**
     * \brief for debug only, should be deleted before submit
     */
    void savePositions(std::ofstream& file, int frame_no) const {
        file << "F " << frame_no << "\n";
        for (const auto& p : _particles) {
            file << p._cur_position.x << " " << p._cur_position.y <<" "<< p._radius << "\n";
        }
    }
};
