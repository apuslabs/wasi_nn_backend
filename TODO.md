# wasi_nn_backend Enhancement Plan

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
- [ ] Extend config parsing to support all sampling parameters
- [ ] Implement advanced sampling algorithms
- [ ] Add stopping criteria configuration
- [ ] Implement memory management settings
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

1. **Phase 1:** Enhanced config parsing (2 days)
2. **Phase 2:** Concurrency and task management (3 days)
3. **Phase 3:** Memory management (2 days)
4. **Phase 4:** Advanced features (3 days)
5. **Phase 5:** Testing and validation (2 days)

## Total Estimated Time: 12 days

## Backward Compatibility Assurance

All existing code using the current API will continue to work without any changes:
- Existing `test/main.c` will run unchanged
- All current function signatures remain the same
- Old configuration JSON structures still work
- No breaking changes to public interfaces
