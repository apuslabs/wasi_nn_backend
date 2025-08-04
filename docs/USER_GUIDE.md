# WASI-NN Backend User Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Configuration](#configuration)
5. [Multi-Session Management](#multi-session-management)
6. [Erlang Integration](#erlang-integration)
7. [Memory Management](#memory-management)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Overview

The WASI-NN Backend is a high-performance library that implements the WebAssembly System Interface for Neural Networks (WASI-NN) API for Llama.cpp models. It provides a stable, production-ready solution for running large language model inference in multi-user, multi-session environments.

### Key Features

- **Multi-Session Support**: Concurrent handling of multiple inference sessions
- **Memory Management**: Advanced KV cache optimization and context shifting
- **Production Ready**: Comprehensive error handling and logging
- **Flexible Configuration**: JSON-based configuration with extensive parameters

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake git

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake git

# macOS
brew install cmake git
```

### Building

```bash
# Clone the repository
git clone https://github.com/apuslabs/wasi_nn_backend.git
cd wasi_nn_backend

# Initialize submodules
git submodule update --init --recursive

# Build the library
mkdir build && cd build
cmake ..
make -j$(nproc)

# The shared library will be generated as:
# build/libwasi_nn_backend.so (Linux)
# build/libwasi_nn_backend.dylib (macOS)
```

### Testing

```bash
# Run the test suite
cd test
make
./main model.gguf
```

## Basic Usage

### C/C++ Integration

```c
#include "wasi_nn_llama.h"

int main() {
    void *backend_ctx = NULL;
    graph_execution_context exec_ctx;
    
    // Initialize backend
    wasi_nn_error err = init_backend(&backend_ctx);
    if (err != success) {
        fprintf(stderr, "Failed to initialize backend: %d\n", err);
        return 1;
    }
    
    // Load model
    graph g;
    const char* config = R"({
        "model": {
            "n_ctx": 4096,
            "n_gpu_layers": 32,
            "threads": 8
        },
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    })";
    
    err = load_by_name_with_config(backend_ctx, "model.gguf", 10, 
                                   config, strlen(config), &g);
    if (err != success) {
        fprintf(stderr, "Failed to load model: %d\n", err);
        return 1;
    }
    
    // Initialize execution context
    err = init_execution_context(backend_ctx, g, &exec_ctx);
    if (err != success) {
        fprintf(stderr, "Failed to initialize execution context: %d\n", err);
        return 1;
    }
    
    // Prepare input tensor
    const char* prompt = "Hello, how are you?";
    tensor input = {0};
    tensor_dimensions dims = {0};
    uint32_t dim_data = strlen(prompt);
    dims.buf = &dim_data;
    dims.size = 1;
    input.dimensions = &dims;
    input.type = u8;
    input.data = (uint8_t*)prompt;
    
    // Set input
    err = set_input(backend_ctx, exec_ctx, 0, &input);
    if (err != success) {
        fprintf(stderr, "Failed to set input: %d\n", err);
        return 1;
    }
    
    // Run inference
    err = compute(backend_ctx, exec_ctx);
    if (err != success) {
        fprintf(stderr, "Failed to compute: %d\n", err);
        return 1;
    }
    
    // Get output
    char output_buffer[4096];
    uint32_t output_size = sizeof(output_buffer);
    err = get_output(backend_ctx, exec_ctx, 0, output_buffer, &output_size);
    if (err == success) {
        printf("Response: %.*s\n", output_size, output_buffer);
    }
    
    // Cleanup
    close_execution_context(backend_ctx, exec_ctx);
    deinit_backend(backend_ctx);
    
    return 0;
}
```

## Configuration

The backend supports comprehensive JSON configuration for fine-tuning behavior. Here's a complete configuration example:

```json
{
  "backend": {
    "max_sessions": 100,
    "idle_timeout_ms": 300000,
    "auto_cleanup": true,
    "max_concurrent": 10,
    "queue_size": 500,
    "default_task_timeout_ms": 30000,
    "priority_scheduling_enabled": true,
    "fair_scheduling_enabled": true,
    "auto_queue_cleanup": true,
    "queue_warning_threshold": 400,
    "queue_reject_threshold": 500
  },
  
  "model": {
    "n_ctx": 4096,
    "n_gpu_layers": 32,
    "batch_size": 512,
    "threads": 8
  },
  
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "typical_p": 1.0,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "penalty_last_n": 64,
    "seed": -1,
    "n_probs": 0,
    "min_keep": 1,
    "ignore_eos": false,
    "grammar": "",
    "grammar_lazy": false,
    
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "dry_sequence_breakers": ["\n", ":", "\"", "*"],
    
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1
  },
  
  "stopping": {
    "max_tokens": 512,
    "ignore_eos": false,
    "stop": ["</s>", "[INST]", "[/INST]"]
  },
  
  "memory": {
    "context_shifting": true,
    "cache_strategy": "lru",
    "max_cache_tokens": 100000,
    "memory_pressure_threshold": 0.8,
    "n_keep_tokens": 128,
    "n_discard_tokens": 256,
    "enable_partial_cache_deletion": true,
    "enable_token_cache_reuse": true,
    "cache_deletion_strategy": "lru",
    "max_memory_mb": 8192
  },
  
  "logging": {
    "level": "info",
    "enable_debug": false,
    "timestamps": true,
    "colors": true,
    "file": "/var/log/wasi_nn_backend.log"
  },
  
  "performance": {
    "batch_processing": true,
    "batch_size": 512
  },
  
  "logit_bias": [
    [198, -1.0],
    [628, 0.5]
  ]
}
```

## Multi-Session Management

The backend is designed for concurrent multi-session usage, making it ideal for server applications:

### Session Lifecycle

```c
// Session management example
typedef struct {
    graph_execution_context ctx;
    char session_id[64];
    time_t last_activity;
    int user_id;
} UserSession;

UserSession sessions[MAX_SESSIONS];
int session_count = 0;

// Create a new session
int create_session(void *backend_ctx, int user_id, const char* session_id) {
    if (session_count >= MAX_SESSIONS) {
        return -1; // Too many sessions
    }
    
    graph_execution_context exec_ctx;
    wasi_nn_error err = init_execution_context(backend_ctx, g, &exec_ctx);
    if (err != success) {
        return -1;
    }
    
    UserSession *session = &sessions[session_count++];
    session->ctx = exec_ctx;
    strncpy(session->session_id, session_id, sizeof(session->session_id) - 1);
    session->last_activity = time(NULL);
    session->user_id = user_id;
    
    return session_count - 1; // Return session index
}

// Process request for a session
int process_session_request(void *backend_ctx, int session_idx, 
                           const char* input, char* output, size_t output_size) {
    if (session_idx < 0 || session_idx >= session_count) {
        return -1;
    }
    
    UserSession *session = &sessions[session_idx];
    session->last_activity = time(NULL);
    
    // Prepare input tensor
    tensor input_tensor = {0};
    tensor_dimensions dims = {0};
    uint32_t dim_data = strlen(input);
    dims.buf = &dim_data;
    dims.size = 1;
    input_tensor.dimensions = &dims;
    input_tensor.type = u8;
    input_tensor.data = (uint8_t*)input;
    
    // Run inference
    wasi_nn_error err = set_input(backend_ctx, session->ctx, 0, &input_tensor);
    if (err != success) return -1;
    
    err = compute(backend_ctx, session->ctx);
    if (err != success) return -1;
    
    uint32_t actual_output_size = output_size;
    err = get_output(backend_ctx, session->ctx, 0, output, &actual_output_size);
    
    return (err == success) ? actual_output_size : -1;
}

// Cleanup idle sessions
void cleanup_idle_sessions(void *backend_ctx, int timeout_seconds) {
    time_t now = time(NULL);
    
    for (int i = 0; i < session_count; i++) {
        if (now - sessions[i].last_activity > timeout_seconds) {
            close_execution_context(backend_ctx, sessions[i].ctx);
            
            // Remove session by shifting array
            for (int j = i; j < session_count - 1; j++) {
                sessions[j] = sessions[j + 1];
            }
            session_count--;
            i--; // Adjust index after removal
        }
    }
}
```

### Thread Safety

The backend is designed to be thread-safe for multi-session scenarios:

```c
#include <pthread.h>

static pthread_mutex_t session_mutex = PTHREAD_MUTEX_INITIALIZER;
static void *global_backend_ctx = NULL;

// Thread-safe session creation
int thread_safe_create_session(int user_id, const char* session_id) {
    pthread_mutex_lock(&session_mutex);
    int result = create_session(global_backend_ctx, user_id, session_id);
    pthread_mutex_unlock(&session_mutex);
    return result;
}

// Thread-safe inference
int thread_safe_inference(int session_idx, const char* input, 
                         char* output, size_t output_size) {
    // Note: Individual sessions are isolated and can run concurrently
    return process_session_request(global_backend_ctx, session_idx, 
                                 input, output, output_size);
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check file permissions
   - Verify model format (GGUF)
   - Ensure sufficient memory

2. **Memory Issues**
   - Reduce context size
   - Enable context shifting
   - Monitor memory usage

3. **Performance Issues**
   - Increase GPU layers
   - Optimize thread count
   - Use appropriate batch size

### Logging Configuration

```json
{
  "logging": {
    "level": "debug",                    // debug, info, warn, error
    "enable_debug": true,                // Enable debug logging
    "timestamps": true,                  // Include timestamps
    "colors": true,                      // Enable colors
    "file": "/var/log/wasi_nn_backend.log" // Log file path
  }
}
```