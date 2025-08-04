# Compiler for test suite
CC_TEST = gcc
CXX = g++

# Test compiler flags - separate from main project
CFLAGS_TEST = -Wall -Wextra -std=c99 -g -O2
LDFLAGS_TEST = -ldl -lpthread

# Compiler flags for main project
CXXFLAGS = -Wall -Wextra -Wformat -Wformat-security -ffunction-sections -fdata-sections -Wno-unused-parameter -fvisibility=hidden -DNN_LOG_LEVEL=0

# Linker flags for main project
LDFLAGS = -ldl

# Test source files
TEST_SRCS = test/main.c
TEST_OBJS = $(TEST_SRCS:.c=.o)
TEST_TARGET = test/main

# Main project files (if any)
SRCS = 
OBJS = $(SRCS:.cpp=.o)
TARGET = 

# Default target - build main project
all: build

# Build the main WASI-NN backend library
build:
	@echo "Building WASI-NN backend library..."
	@if [ ! -d "build" ]; then mkdir -p build; fi
	cd build && cmake .. && $(MAKE) -j16
	@echo "✅ WASI-NN backend library built successfully"

# Build test executable
$(TEST_TARGET): $(TEST_OBJS)
	$(CC_TEST) $(CFLAGS_TEST) -o $@ $^ $(LDFLAGS_TEST)
	@echo "✅ Test executable built successfully"

# Rule to compile test source files
test/%.o: test/%.c
	$(CC_TEST) $(CFLAGS_TEST) -c $< -o $@

# Test rule - build backend, build tests, and run tests
test: build $(TEST_TARGET)
	@echo "============================================================"
	@echo "🚀 Running WASI-NN Backend Test Suite"
	@echo "============================================================"
	@if [ ! -f "build/libwasi_nn_backend.so" ]; then \
		echo "❌ Backend library not found. Build failed."; \
		exit 1; \
	fi
	@echo "✅ Backend library found: build/libwasi_nn_backend.so"
	@echo "✅ Test executable ready: $(TEST_TARGET)"
	@echo "🏃 Executing test suite..."
	@echo "============================================================"
	./$(TEST_TARGET)

# Clean rule - clean both main project and tests
clean:
	rm -f $(TEST_OBJS) $(TEST_TARGET)
	rm -rf build
	@echo "✅ Cleaned all build artifacts"

# Clean only test files
clean-test:
	rm -f $(TEST_OBJS) $(TEST_TARGET)
	@echo "✅ Cleaned test artifacts"

# Install dependencies (if needed)
install-deps:
	@echo "Installing test dependencies..."
	@# Add any dependency installation commands here if needed
	@echo "✅ Dependencies ready"

# Help target
help:
	@echo "Available targets:"
	@echo "  build      - Build the WASI-NN backend library"
	@echo "  test       - Build and run the complete test suite"
	@echo "  clean      - Clean all build artifacts"
	@echo "  clean-test - Clean only test artifacts"
	@echo "  help       - Show this help message"

.PHONY: all build test clean clean-test install-deps help
