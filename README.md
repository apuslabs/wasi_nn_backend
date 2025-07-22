# Llama.cpp Server C API

High-performance C dynamic library based on llama.cpp server implementation, providing clean API interfaces for other C programs.

## Design Goals

- **High Concurrency Support**: Based on llama.cpp server's slot mechanism, supporting multi-user concurrent inference
- **Complete Functionality**: Support for completion, embedding, rerank, streaming and other task types
- **Clean Interface**: Provides clean C interface suitable for dynamic library calls
- **Resource Management**: Efficient memory and GPU resource management
- **NIF Friendly**: Interface design considers Erlang NIF calling limitations

## API Interface Design

### Error Code Definitions
```c
typedef enum {
    LLAMA_SUCCESS = 0,
    LLAMA_ERROR_INVALID_ARGUMENT = 1,
    LLAMA_ERROR_OUT_OF_MEMORY = 2,
    LLAMA_ERROR_MODEL_LOAD_FAILED = 3,
    LLAMA_ERROR_INFERENCE_FAILED = 4,
    LLAMA_ERROR_SESSION_NOT_FOUND = 5,
    LLAMA_ERROR_SERVER_BUSY = 6,
    LLAMA_ERROR_TIMEOUT = 7,
    LLAMA_ERROR_UNKNOWN = 99
} llama_error_t;
```

### Server Management Interface

#### 1. Initialize Server
```c
/**
 * Initialize llama server instance
 * @param server_handle [out] Server handle
 * @param config_json [in] JSON configuration string, can be NULL for default config
 * @return Error code
 */
llama_error_t llama_server_init(void** server_handle, const char* config_json);
```

#### 2. Load Model
```c
/**
 * Load model to server
 * @param server_handle [in] Server handle
 * @param model_path [in] Model file path
 * @param model_config_json [in] Model configuration JSON, can be NULL
 * @return Error code
 */
llama_error_t llama_server_load_model(void* server_handle, 
                                     const char* model_path,
                                     const char* model_config_json);
```

#### 3. Destroy Server
```c
/**
 * Destroy server instance and release all resources
 * @param server_handle [in] Server handle
 * @return Error code
 */
llama_error_t llama_server_destroy(void* server_handle);
```

### Session Management Interface (NEW)

#### 4. Create Session
```c
/**
 * Create a new inference session (maps to llama.cpp slot)
 * @param server_handle [in] Server handle
 * @param session_config [in] Session configuration JSON, can be NULL
 * @param session_id [out] Session ID for this user/conversation
 * @return Error code
 */
llama_error_t llama_server_create_session(void* server_handle,
                                         const char* session_config,
                                         int* session_id);
```

#### 5. Close Session
```c
/**
 * Close and cleanup a session
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID
 * @return Error code
 */
llama_error_t llama_server_close_session(void* server_handle, int session_id);
```

### Inference Interface

#### 6. Create Inference Task
```c
/**
 * Create inference task within a session
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID (for conversation context)
 * @param task_type [in] Task type: "completion", "chat", "embedding", "rerank"
 * @param input_json [in] Input JSON string
 * @param task_id [out] Task ID
 * @return Error code
 */
llama_error_t llama_server_create_task(void* server_handle,
                                      int session_id,
                                      const char* task_type,
                                      const char* input_json,
                                      int* task_id);
```

#### 7. Get Inference Result
```c
/**
 * Get inference result (blocking call)
 * @param server_handle [in] Server handle
 * @param task_id [in] Task ID
 * @param result_buffer [out] Result buffer
 * @param buffer_size [in] Buffer size
 * @param actual_size [out] Actual result size
 * @param timeout_ms [in] Timeout in milliseconds, 0 means infinite wait
 * @return Error code
 */
llama_error_t llama_server_get_result(void* server_handle,
                                     int task_id,
                                     char* result_buffer,
                                     size_t buffer_size,
                                     size_t* actual_size,
                                     int timeout_ms);
```

#### 8. Streaming Inference Interface
```c
/**
 * Streaming inference callback function type
 * @param chunk [in] Current data chunk
 * @param chunk_size [in] Chunk size
 * @param is_final [in] Whether this is the final chunk
 * @param user_data [in] User data
 * @return 0 to continue, non-zero to stop
 */
typedef int (*llama_stream_callback_t)(const char* chunk, size_t chunk_size, 
                                      int is_final, void* user_data);

/**
 * Streaming inference within a session
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID
 * @param input_json [in] Input JSON string
 * @param callback [in] Streaming callback function
 * @param user_data [in] User data pointer
 * @return Error code
 */
llama_error_t llama_server_stream_inference(void* server_handle,
                                           int session_id,
                                           const char* input_json,
                                           llama_stream_callback_t callback,
                                           void* user_data);
```

#### 9. Cancel Task
```c
/**
 * Cancel running task
 * @param server_handle [in] Server handle
 * @param task_id [in] Task ID
 * @return Error code
 */
llama_error_t llama_server_cancel_task(void* server_handle, int task_id);
```

### Status Query Interface

#### 10. Get Server Status
```c
/**
 * Get server status including slot utilization
 * @param server_handle [in] Server handle
 * @param status_buffer [out] Status JSON buffer
 * @param buffer_size [in] Buffer size
 * @param actual_size [out] Actual status size
 * @return Error code
 */
llama_error_t llama_server_get_status(void* server_handle,
                                     char* status_buffer,
                                     size_t buffer_size,
                                     size_t* actual_size);
```

#### 11. Get Session Info
```c
/**
 * Get session information and statistics
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID
 * @param info_buffer [out] Session info JSON buffer
 * @param buffer_size [in] Buffer size
 * @param actual_size [out] Actual info size
 * @return Error code
 */
llama_error_t llama_server_get_session_info(void* server_handle,
                                           int session_id,
                                           char* info_buffer,
                                           size_t buffer_size,
                                           size_t* actual_size);
```

## Configuration Format

### Server Configuration (config_json)
```json
{
    "n_parallel": 4,           // Number of parallel inference slots (key for concurrency)
    "n_ctx": 4096,            // Context size per slot
    "n_batch": 512,           // Batch processing size
    "n_threads": 8,           // CPU threads for computation
    "n_gpu_layers": 35,       // GPU acceleration layers
    "timeout_read": 600,      // Read timeout in seconds
    "timeout_write": 600,     // Write timeout in seconds
    "slot_prompt_similarity": 0.0,  // Prompt similarity threshold for slot reuse
    "cont_batching": true,    // Enable continuous batching for better throughput
    "flash_attn": true        // Enable flash attention if supported
}
```

### Session Configuration (session_config)
```json
{
    "cache_prompt": true,     // Cache prompt tokens for faster subsequent requests
    "system_prompt": "You are a helpful assistant.",  // System prompt for chat sessions
    "n_keep": 10,            // Number of tokens to keep in context
    "temperature": 0.7,      // Default sampling temperature for this session
    "top_p": 0.95           // Default top-p sampling for this session
}
```

### Inference Input Format (input_json)

#### Completion Task
```json
{
    "prompt": "Hello, how are you?",
    "n_predict": 128,
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": false,
    "stop": ["</s>", "\n\n"]
}
```

#### Chat Task
```json
{
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "n_predict": 128,
    "temperature": 0.7,
    "stream": false
}
```

#### Embedding Task
```json
{
    "content": "Text to embed",
    "encoding_format": "float"
}
```

## Build Instructions

### Dependencies
- CUDA (optional, for GPU acceleration)
- CMake >= 3.13
- C++17 compiler

### Build Steps
```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. Build
make -j$(nproc)

# 4. Install (optional)
sudo make install
```

### Output
- `libllama_server.so` - Main dynamic library
- `llama_server.h` - C header file

## Usage Example

### Basic Single Session Example
```c
#include "llama_server.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    void* server = NULL;
    llama_error_t err;
    
    // 1. Initialize server with 4 parallel slots
    const char* config = "{\"n_parallel\": 4, \"n_ctx\": 2048}";
    err = llama_server_init(&server, config);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to init server: %d\n", err);
        return 1;
    }
    
    // 2. Load model
    err = llama_server_load_model(server, "/path/to/model.gguf", NULL);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to load model: %d\n", err);
        llama_server_destroy(server);
        return 1;
    }
    
    // 3. Create a session
    int session_id;
    const char* session_config = "{\"cache_prompt\": true, \"temperature\": 0.7}";
    err = llama_server_create_session(server, session_config, &session_id);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to create session: %d\n", err);
        llama_server_destroy(server);
        return 1;
    }
    
    // 4. Create inference task
    const char* input = "{\"prompt\": \"Hello\", \"n_predict\": 50}";
    int task_id;
    err = llama_server_create_task(server, session_id, "completion", input, &task_id);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to create task: %d\n", err);
        llama_server_close_session(server, session_id);
        llama_server_destroy(server);
        return 1;
    }
    
    // 5. Get result
    char result[4096];
    size_t result_size;
    err = llama_server_get_result(server, task_id, result, sizeof(result), 
                                 &result_size, 10000);
    if (err == LLAMA_SUCCESS) {
        printf("Result: %s\n", result);
    }
    
    // 6. Cleanup
    llama_server_close_session(server, session_id);
    llama_server_destroy(server);
    return 0;
}
```

### Concurrent Multi-Session Example
```c
#include "llama_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    void* server;
    int session_id;
    int thread_id;
} thread_data_t;

void* worker_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    // Create inference task in this session
    char input[256];
    snprintf(input, sizeof(input), 
             "{\"prompt\": \"Hello from thread %d\", \"n_predict\": 30}", 
             data->thread_id);
    
    int task_id;
    llama_error_t err = llama_server_create_task(data->server, data->session_id, 
                                                "completion", input, &task_id);
    if (err != LLAMA_SUCCESS) {
        printf("Thread %d: Failed to create task\n", data->thread_id);
        return NULL;
    }
    
    // Get result
    char result[1024];
    size_t result_size;
    err = llama_server_get_result(data->server, task_id, result, sizeof(result), 
                                 &result_size, 30000);
    if (err == LLAMA_SUCCESS) {
        printf("Thread %d result: %s\n", data->thread_id, result);
    } else {
        printf("Thread %d: Failed to get result: %d\n", data->thread_id, err);
    }
    
    return NULL;
}

int main() {
    void* server = NULL;
    llama_error_t err;
    
    // Initialize server with 4 parallel slots for concurrency
    const char* config = "{\"n_parallel\": 4, \"n_ctx\": 2048, \"cont_batching\": true}";
    err = llama_server_init(&server, config);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to init server: %d\n", err);
        return 1;
    }
    
    // Load model
    err = llama_server_load_model(server, "/path/to/model.gguf", NULL);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to load model: %d\n", err);
        llama_server_destroy(server);
        return 1;
    }
    
    // Create multiple sessions and threads
    const int num_threads = 4;
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];
    
    // Create sessions for each thread
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].server = server;
        thread_data[i].thread_id = i;
        
        err = llama_server_create_session(server, NULL, &thread_data[i].session_id);
        if (err != LLAMA_SUCCESS) {
            printf("Failed to create session %d: %d\n", i, err);
            // Cleanup and exit
            for (int j = 0; j < i; j++) {
                llama_server_close_session(server, thread_data[j].session_id);
            }
            llama_server_destroy(server);
            return 1;
        }
    }
    
    // Start concurrent inference
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Cleanup all sessions
    for (int i = 0; i < num_threads; i++) {
        llama_server_close_session(server, thread_data[i].session_id);
    }
    
    llama_server_destroy(server);
    return 0;
}
```

## NIF Integration Notes

1. **Simple Data Types**: All interfaces use basic C data types (int, char*, size_t, etc.)
2. **JSON Communication**: Complex data passed through JSON strings for easy NIF parsing
3. **Error Handling**: Unified error code system for convenient Erlang error handling
4. **Memory Management**: Caller responsible for providing buffers, avoiding dynamic allocation
5. **Async Support**: Asynchronous operations through task IDs, suitable for Erlang's concurrency model
6. **Session-based Concurrency**: Each Erlang process can maintain its own session_id for isolated conversations
7. **Parallel Processing**: Multiple Erlang processes can create tasks simultaneously, server handles slot allocation automatically

### Erlang Usage Pattern
```erlang
% Initialize server once
{ok, Server} = llama_nif:init_server(Config),

% Each user conversation gets its own session
{ok, SessionId1} = llama_nif:create_session(Server, SessionConfig),
{ok, SessionId2} = llama_nif:create_session(Server, SessionConfig),

% Multiple processes can inference concurrently
spawn(fun() -> 
    {ok, TaskId} = llama_nif:create_task(Server, SessionId1, "chat", Input1),
    {ok, Result} = llama_nif:get_result(Server, TaskId, 30000)
end),

spawn(fun() -> 
    {ok, TaskId} = llama_nif:create_task(Server, SessionId2, "chat", Input2),
    {ok, Result} = llama_nif:get_result(Server, TaskId, 30000)
end).
```