# wasi_nn_backend Enhancement Plan

## Updated Plan with llama.cpp Server Integration

This plan has been updated to integrate the llama.cpp server functionality into the WASI-NN backend, following an iterative approach with testing at each phase.

## Overview
Enhance the wasi_nn_backend implementation to include advanced features while maintaining backward compatibility with existing interfaces.

## Goals
1. Advanced Concurrency and Task Management
2. Model reloading and hot-swapping capabilities
3. Sophisticated Memory Management
4. Comprehensive sampling parameter support
5. Advanced stopping criteria
6. Comprehensive logging
7. Better Error Handling and Validation
8. Performance Optimizations

## Implementation Approach
- Maximize retention of existing interfaces
- Extend functionality through configuration parameters
- Add minimal new APIs only when necessary
- Maintain 100% backward compatibility

## Detailed Implementation Plan

### Phase 1: Enhanced Configuration System

#### 1.1 Enhanced Backend Initialization
**File:** `src/wasi_nn_llama.cpp`
**Function:** `init_backend_with_config`

**Enhanced Config Structure:**
```json
{
  "max_sessions": 100,
  "idle_timeout_ms": 300000,
  "auto_cleanup": true,
  "max_concurrent": 8,
  "queue_size": 50,
  "memory_policy": {
    "context_shifting": true,
    "cache_strategy": "lru",
    "max_cache_tokens": 10000
  },
  "logging": {
    "level": "info",
    "enable_debug": false,
    "file": "/path/to/logfile.log"
  },
  "performance": {
    "batch_processing": true,
    "batch_size": 512
  }
}
```

**Tasks:**
- [x] Extend config parsing in `init_backend_with_config`
- [x] Implement concurrency limit management
- [ ] Add queue management system
- [ ] Implement memory policy configuration
- [ ] Add logging configuration support

#### 1.2 Enhanced Model Loading
**File:** `src/wasi_nn_llama.cpp`
**Function:** `load_by_name_with_config`

**Enhanced Model Config Structure:**
```json
{
  "n_gpu_layers": 98,
  "ctx_size": 2048,
  "n_predict": 512,
  "batch_size": 512,
  "threads": 8,
  "sampling": {
    "temp": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "tfs_z": 1.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.10,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "penalty_last_n": 64,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "seed": -1
  },
  "stopping": {
    "stop": ["\n", "User:", "Assistant:"],
    "max_tokens": 512,
    "max_time_ms": 30000,
    "ignore_eos": false
  },
  "memory": {
    "context_shifting": true,
    "cache_prompt": true
  }
}
```

**Tasks:**
- [x] Extend config parsing to support all sampling parameters
- [ ] Implement advanced sampling algorithms
- [x] Add stopping criteria configuration
- [x] Implement memory management settings
- [ ] Add model hot-swapping capability

### Phase 2: Concurrency and Task Management

#### 2.1 Automatic Task Queuing
**File:** `src/wasi_nn_llama.cpp`
**Function:** `run_inference`

**Tasks:**
- [ ] Implement internal task queue system
- [ ] Add automatic queuing when resources are busy
- [ ] Implement priority queuing for fairness
- [ ] Add queue size limiting
- [ ] Implement task timeout handling

#### 2.2 Unified Status Query API
**File:** `src/wasi_nn_llama.h`, `src/wasi_nn_llama.cpp`

**New API:**
```c
typedef enum {
    BACKEND_STATUS_QUEUE,
    BACKEND_STATUS_MEMORY,
    BACKEND_STATUS_SESSIONS,
    BACKEND_STATUS_PERFORMANCE,
    BACKEND_STATUS_MODEL
} backend_status_type;

wasi_nn_error get_backend_status(void *ctx, backend_status_type type, char *buffer, uint32_t buffer_size);
```

**Tasks:**
- [ ] Add `get_backend_status` function declaration to header
- [ ] Implement queue status reporting
- [ ] Implement memory status reporting
- [ ] Implement session status reporting
- [ ] Implement performance status reporting
- [ ] Implement model status reporting

### Phase 3: Memory Management

#### 3.1 Context Shifting
**Tasks:**
- [ ] Implement context shifting for long conversations
- [ ] Add partial KV cache deletion
- [ ] Implement token cache reuse mechanisms
- [ ] Add memory pressure handling

#### 3.2 Enhanced Session Management
**File:** `src/wasi_nn_llama.cpp`

**Enhanced Session Config:**
```json
{
  "stopping": {
    "stop": ["\n\n"],
    "max_tokens": 256,
    "max_time_ms": 15000
  },
  "sampling_override": {
    "temp": 0.9
  }
}
```

**Tasks:**
- [ ] Implement `init_execution_context_with_config`
- [ ] Add session-level stopping criteria
- [ ] Add session-level sampling overrides
- [ ] Implement session resource tracking

### Phase 4: Advanced Features

#### 4.1 Comprehensive Logging
**Tasks:**
- [ ] Implement multi-level logging (DEBUG, INFO, WARN, ERROR)
- [ ] Add structured logging with metadata
- [ ] Implement log filtering capabilities
- [ ] Add file-based logging support
- [ ] Add performance logging

#### 4.2 Performance Optimizations
**Tasks:**
- [ ] Implement token batch processing
- [ ] Add performance metrics collection
- [ ] Create caching strategies
- [ ] Add benchmarking capabilities

### Phase 5: Testing and Validation

#### 5.1 Backward Compatibility Testing
**Tasks:**
- [ ] Verify existing test code still works
- [ ] Test all existing API functions
- [ ] Validate config parsing backward compatibility

#### 5.2 New Feature Testing
**Tasks:**
- [ ] Test concurrency and queuing
- [ ] Test model hot-swapping
- [ ] Test advanced sampling parameters
- [ ] Test stopping criteria
- [ ] Test memory management features
- [ ] Test status reporting

## Interface Changes Summary

### Core APIs (No Changes)
- `init_backend`
- `deinit_backend`
- `load_by_name_with_config`
- `init_execution_context`
- `close_execution_context`
- `run_inference`

### Extended APIs (Backward Compatible)
- `init_backend_with_config` (enhanced config)
- `init_execution_context_with_config` (new function)
- `get_backend_status` (new function)

## Implementation Priority

## Implementation Status

### Phase 1: Integration Preparation - COMPLETED ✓
Server.cpp integration and build system setup completed successfully.

### Phase 2: Core Integration - COMPLETED ✓  
Core integration successfully completed. Model loading, backend initialization, and WASI-NN interface bridge are working.

**Key Achievement**: Resolved symbol linking issues by directly including server.cpp in wasi_nn_llama.cpp.

**Current Issue**: Runtime segfault during inference needs debugging (Phase 3 task).

### Phase 3: Debug and Optimize Inference - IN PROGRESS
**Next Priority**: Fix segmentation fault during inference execution.

**Current working state**:
- ✅ Backend initialization 
- ✅ Model loading (14B Qwen2.5 successfully loads to GPU)
- ✅ Execution context creation
- ❌ Inference execution (segfaults during token generation)

**Debugging needed**:
1. Server slots initialization and sampler setup
2. Chat template and tokenization memory handling  
3. Batch processing configuration

### Phase 3: Feature Implementation (Iterative)
1. Implement enhanced configuration system
2. Test configuration parsing - Verify all sampling parameters work
3. Add advanced concurrency and task management
4. Test concurrency - Verify multiple sessions work correctly
5. Implement model reloading and hot-swapping capabilities
6. Test model switching - Verify models can be swapped without crashes
7. Add sophisticated memory management
8. Test memory handling - Verify stability under memory pressure
9. Implement comprehensive sampling parameter support
10. Test sampling parameters - Verify all sampling methods work correctly
11. Add advanced stopping criteria
12. Test stopping criteria - Verify all stopping conditions work

### Phase 4: Final Integration and Validation
1. Update all remaining TODO items
2. Comprehensive testing of all features
3. Performance optimization
4. Final validation of backward compatibility

## Total Estimated Time: 12 days

## Backward Compatibility Assurance

All existing code using the current API will continue to work without any changes:
- Existing `test/main.c` will run unchanged
- All current function signatures remain the same
- Old configuration JSON structures still work
- No breaking changes to public interfaces

## Development Approach

This implementation will follow an iterative development process with testing at each step:

1. **Early Validation**: We'll have a working system after Phase 2 that you can test
2. **Incremental Progress**: Each step builds on the previous one with immediate testing
3. **Risk Mitigation**: Issues are caught early before they become complex problems
4. **Collaborative Development**: You can review and provide feedback at each milestone

For each step, I will:
1. Implement the code changes
2. Update the TODO.md to reflect completed work
3. Stop and wait for your review and git commit before proceeding

## Interface Compatibility Principles

1. **Preserve Existing Interfaces**: All current function signatures in wasi_nn_llama.h will remain unchanged
2. **Extend via Configuration**: New functionality will be accessed through enhanced JSON configuration parameters
3. **Backward Compatibility**: Old configuration JSON structures will continue to work
4. **Minimal New APIs**: Only add new functions when absolutely necessary, and only as extensions
