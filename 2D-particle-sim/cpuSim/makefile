# Compiler and flags
CXX      = g++
CXXFLAGS = -std=c++17 -Wall -O3 -I/opt/homebrew/Cellar/sfml/3.0.0_1/include

# Linker flags and libraries
LDFLAGS   := -L/opt/homebrew/Cellar/sfml/3.0.0_1/lib
LDLIBS    := -lsfml-graphics -lsfml-window -lsfml-system

# Target executable name
TARGET   = simulator

# Source and object files
SRCS     = main.cpp simulator.cpp renderer.cpp
OBJS     = $(SRCS:.cpp=.o)

# Default target: compile the executable
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $(TARGET)

# Compile each .cpp file into an object file
%.o: %.cpp simulator.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target: remove executable and object files
clean:
	rm -f $(TARGET) $(OBJS)

# Test target: build and run the executable
test: $(TARGET)
	./$(TARGET) 100

# Declare phony targets
.PHONY: all clean test
