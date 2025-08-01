# wasi_nn_backend

A WASI-NN backend implementation for Llama.cpp models, enabling WebAssembly modules to perform inference using large language models.

## Project Overview

This project provides a shared library that implements the WASI-NN (WebAssembly System Interface for Neural Networks) API for Llama.cpp models. It allows WebAssembly modules to load and run inference on quantized GGUF models with GPU acceleration support.

## Project Structure

```
wasi_nn_backend/
├── src/                          # Core implementation
│   ├── wasi_nn_llama.h          # WASI-NN API declarations
│   ├── wasi_nn_llama.cpp        # Main implementation
│   └── utils/
│       └── logger.h             # Logging utilities
├── lib/
│   └── llama.cpp/               # Llama.cpp submodule
│       ├── src/                 # Core llama.cpp source
│       ├── common/              # Common utilities and structures
│       └── tools/server/        # Reference server implementation
├── test/                        # Test files and models
│   ├── main.c                   # Test program
│   ├── *.gguf                   # Test model files
│   └── Makefile                 # Test build configuration
├── build/                       # Build output directory
│   ├── libwasi_nn_backend.so    # Generated shared library
│   └── CMakeFiles/              # CMake build files
├── CMakeLists.txt               # Main CMake configuration
├── README.md                    # This file
└── TODO.md                      # Project status and roadmap
```

### Key Files and Their Roles

#### **Core Implementation (`src/`)**
- **`wasi_nn_llama.h`**: Defines the WASI-NN API interface, data structures, and function declarations
- **`wasi_nn_llama.cpp`**: Main implementation file containing all WASI-NN functions, model management, and inference logic
- **`utils/logger.h`**: Comprehensive logging system with multiple levels and structured output

#### **Reference Implementation (`lib/llama.cpp/`)**
- **`tools/server/server.cpp`**: Gold standard reference for parameter handling, validation, and best practices
- **`common/`**: Shared utilities, parameter structures, and helper functions
- **`src/`**: Core llama.cpp inference engine

#### **Testing and Validation (`test/`)**
- **`main.c`**: Comprehensive test suite covering all implemented features
- **`*.gguf`**: Test model files for validation (Qwen2.5-14B, Phi3-3B)
- **`Makefile`**: Independent test build system

#### **Build System**
- **`CMakeLists.txt`**: Main build configuration with CUDA support and optimization flags
- **`build/`**: Contains generated shared library and build artifacts

## Project Architecture and Design Principles

### Core Design Philosophy

This WASI-NN backend is designed with **production-grade reliability** and **reference-driven quality** as primary goals:

#### **1. Reference-Driven Development**
- **Gold Standard**: `lib/llama.cpp/tools/server/server.cpp` serves as the reference implementation
- **Parameter Completeness**: All quality-enhancing parameters from server.cpp are targeted for inclusion
- **Validation Depth**: Match server.cpp's comprehensive parameter validation and error handling
- **Memory Management**: Implement server.cpp's intelligent resource management strategies

#### **2. Backward Compatibility First**
- **API Stability**: All existing function signatures remain unchanged
- **Configuration Compatibility**: Old JSON configurations continue to work
- **Zero Breaking Changes**: Existing code requires no modifications
- **Progressive Enhancement**: New features are additive, not replacements

#### **3. Production-Grade Robustness**
- **Defensive Programming**: Comprehensive validation for all inputs and states
- **Graceful Degradation**: System continues operating even when individual components fail
- **Resource Safety**: Automatic cleanup and leak prevention
- **Signal Handling**: Protection against edge cases and system interrupts

#### **4. Performance and Quality Balance**
- **Quality Parameters Only**: Only implement parameters that measurably improve generation quality
- **Intelligent Defaults**: Complex parameters have user-friendly default values
- **Automatic Optimization**: Memory management and resource allocation work transparently
- **Minimal Complexity**: Advanced features don't complicate basic usage

### Development Phases Overview

The project follows a structured 7-phase development approach:

**Phases 1-3**: Foundation (Integration, Stability, Core Features) ✅  
**Phases 4-6**: Advanced Features (Concurrency, Memory, Logging, Stopping) ✅  
**Phase 7**: Quality Optimization (Enhanced Sampling, Validation, Error Handling) 📋

Each phase builds upon previous work while maintaining full backward compatibility and comprehensive testing.

### Key Features

- WASI-NN API compliance
- Llama.cpp integration with GPU (CUDA) support
- Advanced session management with task queuing and priority handling
- Support for quantized GGUF models with automatic optimization
- Comprehensive logging system with structured output
- Model hot-swapping capabilities without service interruption
- Advanced stopping criteria with grammar triggers and semantic detection
- Automatic memory management with KV cache optimization and context shifting

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
# Build project first
cd build & make -j16 & cd ..
# Build and run the test & Run the test executable
make test & ./main
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

## Troubleshooting

### Common Issues and Solutions

1. **CUDA initialization errors**: Ensure you have a compatible NVIDIA GPU and CUDA drivers installed.
2. **Model loading failures**: Verify the model file path and format (should be GGUF).
3. **Memory allocation errors**: Large models may require significant GPU memory.

### Development Philosophy: Reference-Driven Enhancement

This project follows a **reference-driven development approach** using `lib/llama.cpp/tools/server/server.cpp` as the gold standard for:

#### **Parameter Completeness**
**Goal**: Match server.cpp's comprehensive parameter support
- **Current Gap**: Missing advanced sampling parameters (dynatemp, DRY suppression)
- **Target**: Implement all quality-enhancing parameters from server.cpp
- **Rationale**: Server.cpp represents the most mature and tested parameter set

#### **Validation Robustness** 
**Goal**: Match server.cpp's parameter validation depth
- **Current Gap**: Basic range checking vs comprehensive validation
- **Target**: Implement server.cpp's validation logic including cross-parameter dependencies
- **Example**: Automatic `penalty_last_n = -1` → `penalty_last_n = ctx_size` adjustment

#### **Error Handling Quality**
**Goal**: Match server.cpp's detailed error reporting
```cpp
// Current (basic):
LOG_ERROR("Failed to load model");

// Target (server.cpp style):
LOG_ERROR("Failed to load model '%s': %s\n"
          "Suggestion: Check file path and permissions\n" 
          "Available memory: %.2f GB, Required: %.2f GB",
          model_path, error_detail, avail_mem, req_mem);
```

#### **Memory Management Excellence**
**Goal**: Implement server.cpp's intelligent resource management
- **Dynamic Resource Allocation**: Auto-adjust GPU layers, batch size, context size
- **Intelligent Cache Management**: Prompt caching, KV cache reuse, similarity-based sharing
- **Memory Pressure Handling**: Automatic cleanup and optimization under memory constraints

#### **Configuration System Sophistication**
**Goal**: Support server.cpp's nested configuration structure
```json
{
  "sampling": {
    "dynatemp": {"range": 0.1, "exponent": 1.2}
  },
  "memory_management": {
    "cache_policy": {"enable_prompt_cache": true, "retention_ms": 300000}
  }
}
```

### Debugging

Enable debug logging by setting the appropriate log level during compilation.

## Development Best Practices

### Critical Development Lessons

Based on extensive debugging and development experience, follow these practices:

#### **1. Always Initialize Before Use**
```cpp
// CRITICAL: Check and initialize all components
if (!slot.smpl) {
    slot.smpl = common_sampler_init(model, params.sampling);
}
if (!has_chat_template) {
    apply_chat_template_or_fallback();
}
```

#### **2. Implement Comprehensive Validation**
```cpp
// Validate ALL parameters before use
static wasi_nn_error validate_params(const common_params_sampling& params) {
    if (params.temp < 0.0f || params.temp > 2.0f) return wasi_nn_error_invalid_argument;
    if (params.top_p < 0.0f || params.top_p > 1.0f) return wasi_nn_error_invalid_argument;
    if (params.penalty_last_n < -1) return wasi_nn_error_invalid_argument;
    return wasi_nn_error_none;
}
```

#### **3. Follow server.cpp Patterns**
When implementing new features, always reference server.cpp for:
- Parameter parsing patterns
- Validation logic
- Error handling approaches
- Memory management strategies
- Configuration structure

#### **4. Defensive Resource Management**
```cpp
// Always clean up in reverse order of allocation
void cleanup_resources() {
    if (sampler) common_sampler_free(sampler);
    if (context) llama_free(context);  
    if (model) llama_free_model(model);
}
```

#### **5. Test Edge Cases Thoroughly**
- Large models (14B+) with limited GPU memory
- Invalid parameter combinations
- Resource exhaustion scenarios
- Concurrent access patterns
- Model switching under load

### Reference Implementation Strategy

#### **server.cpp Analysis Approach**
1. **Parameter Discovery**: Use `grep` to find all parameter handling
2. **Validation Analysis**: Study validation patterns and error messages
3. **Memory Management Review**: Understand resource allocation strategies
4. **Configuration Structure**: Map nested parameter hierarchies
5. **Error Handling Patterns**: Learn from detailed error reporting

#### **Implementation Priority**
1. **Critical Safety**: Parameter validation prevents crashes
2. **Quality Enhancement**: Sampling parameters improve generation
3. **User Experience**: Detailed error messages aid debugging
4. **Performance**: Memory management optimizes resource usage
5. **Professional Polish**: Nested configuration provides flexibility

### Server.cpp Reference Points

Key areas where server.cpp provides valuable reference:

#### **Sampling Parameter Completeness**
- `dynatemp_range`, `dynatemp_exponent` for dynamic temperature control
- `dry_multiplier`, `dry_base` etc. for repetition suppression
- Automatic parameter adjustment (e.g., `penalty_last_n = -1` → `ctx_size`)

#### **Validation Robustness**
- Range checking for all numerical parameters
- Cross-parameter dependency validation
- Automatic value adjustment and normalization

#### **Error Handling Excellence**
- Detailed error messages with specific causes
- Suggestions for problem resolution
- System resource information in error context

#### **Memory Management Intelligence**
- Dynamic GPU layer calculation based on available memory
- Intelligent batch size adjustment
- Cache management with similarity-based reuse

#### **Configuration Sophistication**
- Nested parameter structure support
- Smart default value inheritance
- Deep configuration validation

## Contributing

### Development Guidelines

Contributions are welcome! When contributing:

1. **Follow Reference-Driven Approach**: Use server.cpp as the quality standard
2. **Maintain Backward Compatibility**: Ensure existing code continues to work
3. **Add Comprehensive Tests**: Include tests for all new functionality
4. **Document Critical Issues**: Share debugging insights and solutions
5. **Validate Against Production Workloads**: Test with real-world model sizes

### Reporting Issues

When reporting issues, please include:
- Model size and type (e.g., "Qwen2.5-14B Q4_K_M")
- Configuration parameters used
- System specifications (GPU model, CUDA version, available memory)
- Complete error logs with context

### Pull Request Process

1. Reference server.cpp implementation for similar functionality
2. Add comprehensive parameter validation
3. Include detailed error handling with suggestions
4. Test with multiple model sizes and configurations
5. Update documentation with new capabilities

## License

This project is licensed under the Apache License 2.0 with LLVM Exception.

## Acknowledgments

- **llama.cpp**: Core inference engine and reference implementation
- **WASI-NN**: WebAssembly neural network interface standard
- **Community**: Contributors and testers who helped improve reliability
