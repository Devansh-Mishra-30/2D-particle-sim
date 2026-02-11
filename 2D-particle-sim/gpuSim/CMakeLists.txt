cmake_minimum_required(VERSION 3.18)

project(particle_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

add_executable(particle_sim
  main.cpp
  renderer.cpp
  simulator_cuda.cu
)

target_include_directories(particle_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(particle_sim PRIVATE SFML::Graphics SFML::Window SFML::System)

# Warnings and optimization per build type
target_compile_options(particle_sim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# CUDA settings
set_target_properties(particle_sim PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# If you want explicit arch, set it here.
# For example, sm_89:
# set(CMAKE_CUDA_ARCHITECTURES 89)
# Better: use native on newer CMake/NVIDIA toolchains if supported:
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
