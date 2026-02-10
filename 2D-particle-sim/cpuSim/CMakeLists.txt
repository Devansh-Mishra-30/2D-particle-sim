cmake_minimum_required(VERSION 3.16)
project(particle_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

set(SOURCES
  main.cpp
  simulator.cpp
  renderer.cpp
)

# Common flags for both targets (keeps CPU vs OMP fair)
add_library(sim_common_flags INTERFACE)
target_compile_options(sim_common_flags INTERFACE
  -O3
  -Wall
  -Wextra
  -march=native
)

# -----------------------
# CPU target (no OpenMP)
# -----------------------
add_executable(sim_cpu ${SOURCES})
target_link_libraries(sim_cpu PRIVATE sim_common_flags SFML::Graphics SFML::Window SFML::System)

# -----------------------
# OMP target (OpenMP on)
# -----------------------
find_package(OpenMP REQUIRED)

add_executable(sim_omp ${SOURCES})
target_link_libraries(sim_omp PRIVATE sim_common_flags OpenMP::OpenMP_CXX SFML::Graphics SFML::Window SFML::System)
