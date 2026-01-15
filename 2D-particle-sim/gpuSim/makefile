# Makefile for CUDA particle simulation using SFML 3.0.0 on Linux

# Compilers
NVCC       := nvcc
CXX        := g++

# Flags
NVCC_FLAGS := -std=c++17 -arch=sm_89
CXX_FLAGS  := -std=c++17 -O2

# SFML paths (adjust if needed)
SFML_PATH  := /home/devansh/SFML-3.0.0-linux
SFML_INC   := -I$(SFML_PATH)/include
SFML_LIB   := -L$(SFML_PATH)/lib
LIBS       := -lsfml-graphics -lsfml-window -lsfml-system

# Targets
TARGET     := particle_sim
OBJS       := main.o renderer.o simulator_cuda.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(SFML_LIB) $(LIBS)

main.o: main.cpp renderer.h simulator_cuda.cuh particle_data.h
	$(CXX) $(CXX_FLAGS) $(SFML_INC) -c $< -o $@

renderer.o: renderer.cpp renderer.h particle_data.h
	$(CXX) $(CXX_FLAGS) $(SFML_INC) -c $< -o $@

simulator_cuda.o: simulator_cuda.cu simulator_cuda.cuh particle_data.h
	$(NVCC) $(NVCC_FLAGS) $(SFML_INC) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
