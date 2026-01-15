cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME simulator)
project(${PROJECT_NAME})

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)

# Enable all warnings and set O3 optimizations
set(CMAKE_CXX_FLAGS "-Wall -O3")

# Detect and add SFML
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

# Include directories
include_directories(. ${SFML_INCLUDE_DIRS})

# Source files
set(SOURCES main.cpp simulator.cpp renderer.cpp)

# Specify the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link SFML libraries
target_link_libraries(${PROJECT_NAME} SFML::Graphics SFML::Window SFML::System)
