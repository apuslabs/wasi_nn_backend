# Makefile for WASI-NN Backend Test Suite
# Usage: make test && ./main

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -O2
LDFLAGS = -ldl -lpthread

# Source files
SOURCES = main.c
TARGET = main

# Build target
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Test target
test: $(TARGET)
	@echo "✅ Test executable built successfully"
	@echo "Run with: ./$(TARGET)"

# Clean target
clean:
	rm -f $(TARGET)
	@echo "✅ Cleaned test executable"

# Install target (ensure backend library exists)
install:
	@echo "Checking backend library..."
	@if [ ! -f "../build/libwasi_nn_backend.so" ]; then \
		echo "❌ Backend library not found. Please run 'make -C .. -j16' first"; \
		exit 1; \
	else \
		echo "✅ Backend library found"; \
	fi

# Full build and test
all: install test
	@echo "✅ Ready to run tests with: ./$(TARGET)"

.PHONY: test clean install all
