# wasi_nn_backend

A WASI-NN backend implementation for Llama.cpp models, enabling WebAssembly modules to perform inference using large language models.

## Project Overview

This project provides a shared library that implements the WASI-NN (WebAssembly System Interface for Neural Networks) API for Llama.cpp models. It allows WebAssembly modules to load and run inference on quantized GGUF models with GPU acceleration support.

Key features:
- WASI-NN API compliance
- Llama.cpp integration with GPU (CUDA) support
- Session management for multiple concurrent inferences
- Support for quantized GGUF models
- Automatic session cleanup and resource management

## Prerequisites

- CMake 3.13 or higher
- C++17 compatible compiler
- CUDA toolkit (for GPU acceleration)
- NVIDIA GPU (for CUDA support)

## Building the Project

```bash
# Create build directory
mkdir build

# Navigate to build directory
cd build

# Run CMake
cmake ..

# Build the project
make -j16
```

The build process will create a shared library `libwasi_nn_backend.so` in the build directory.

## Running Tests

The project includes a test program that demonstrates how to use the WASI-NN backend:

```bash
# Build and run the test
make test

# Run the test executable
./main
```

The test program:
1. Loads a quantized GGUF model from the `test` directory
2. Initializes the backend and execution context
3. Runs inference on sample prompts
4. Cleans up resources

## API Documentation

The backend implements the following WASI-NN functions:

### Core Functions

- `init_backend(void **ctx)` - Initialize the backend context
- `load_by_name_with_config(void *ctx, const char *filename, uint32_t filename_len, const char *config, uint32_t config_len, graph *g)` - Load a model with configuration
- `init_execution_context(void *ctx, graph g, graph_execution_context *exec_ctx)` - Initialize an execution context
- `run_inference(void *ctx, graph_execution_context exec_ctx, uint32_t index, tensor *input_tensor, tensor_data output_tensor, uint32_t *output_tensor_size)` - Run inference
- `deinit_backend(void *ctx)` - Deinitialize the backend

### Configuration Options

The `load_by_name_with_config` function accepts a JSON configuration string with the following options:

```json
{
  "n_gpu_layers": 98,
  "ctx_size": 2048,
  "n_predict": 512,
  "batch_size": 512,
  "threads": 8,
  "temp": 0.7,
  "top_p": 0.95,
  "repeat_penalty": 1.10
}
```

### Session Management

The backend supports session management with automatic cleanup:
- Maximum sessions: 100 (configurable)
- Idle timeout: 300000ms (5 minutes, configurable)
- Automatic cleanup of idle sessions

## Model Requirements

The backend supports GGUF format models. Place your quantized GGUF model file in the `test` directory and update the model filename in `test/main.c`.

## Troubleshooting

### Common Issues

1. **CUDA initialization errors**: Ensure you have a compatible NVIDIA GPU and CUDA drivers installed.
2. **Model loading failures**: Verify the model file path and format (should be GGUF).
3. **Memory allocation errors**: Large models may require significant GPU memory.

### Debugging

Enable debug logging by setting the appropriate log level during compilation.

## Contributing

Contributions are welcome! Please submit issues and pull requests to the GitHub repository.

## License

This project is licensed under the Apache License 2.0 with LLVM Exception.
