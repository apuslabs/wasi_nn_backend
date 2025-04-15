# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -Wformat -Wformat-security -ffunction-sections -fdata-sections -Wno-unused-parameter -fvisibility=hidden

# Linker flags
LDFLAGS = -ldl

# Source files
SRCS = test/main.c

# Object files
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = main

# Default target
all: $(TARGET)

# Rule to link the object files into an executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile source files into object files
%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Test rule to compile and execute the program
test: $(TARGET)
	rm -f $(OBJS) 

.PHONY: all clean test