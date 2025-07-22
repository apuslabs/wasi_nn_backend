/*
 * Llama Server C API Test Suite
 */

#include "llama_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

// Test configuration
static const char* test_model_path = "./test/Qwen2.5-1.5B-Instruct.Q2_K.gguf";
static const char* server_config = R"({
    "n_parallel": 4,
    "n_ctx": 2048,
    "n_batch": 512,
    "n_threads": 8,
    "cont_batching": true
})";

static const char* session_config = R"({
    "cache_prompt": true,
    "temperature": 0.7,
    "top_p": 0.95
})";

// Test utilities
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("TEST FAILED: %s\n", message); \
            return -1; \
        } else { \
            printf("PASS: %s\n", message); \
        } \
    } while(0)

#define TEST_FUNCTION(func) \
    do { \
        printf("\n=== Running %s ===\n", #func); \
        if (func() != 0) { \
            printf("TEST SUITE FAILED at %s\n", #func); \
            return -1; \
        } \
    } while(0)

// ================================================================
// Basic API Tests
// ================================================================

int test_server_lifecycle() {
    void* server = NULL;
    llama_error_t err;
    
    // Test initialization
    err = llama_server_init(&server, server_config);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Server initialization");
    TEST_ASSERT(server != NULL, "Server handle is not NULL");
    
    // Test model loading
    err = llama_server_load_model(server, test_model_path, NULL);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Model loading");
    
    // Test server status
    char status[1024];
    size_t status_size;
    err = llama_server_get_status(server, status, sizeof(status), &status_size);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Server status query");
    printf("Server status: %s\n", status);
    
    // Test cleanup
    err = llama_server_destroy(server);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Server cleanup");
    
    return 0;
}

int test_session_management() {
    void* server = NULL;
    llama_error_t err;
    
    // Initialize server
    err = llama_server_init(&server, server_config);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Server initialization");
    
    err = llama_server_load_model(server, test_model_path, NULL);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Model loading");
    
    // Test session creation
    int session_id1, session_id2;
    err = llama_server_create_session(server, session_config, &session_id1);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Session 1 creation");
    
    err = llama_server_create_session(server, session_config, &session_id2);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Session 2 creation");
    
    TEST_ASSERT(session_id1 != session_id2, "Session IDs are unique");
    
    // Test session info
    char info[512];
    size_t info_size;
    err = llama_server_get_session_info(server, session_id1, info, sizeof(info), &info_size);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Session info query");
    
    // Test session cleanup
    err = llama_server_close_session(server, session_id1);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Session 1 cleanup");
    
    err = llama_server_close_session(server, session_id2);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Session 2 cleanup");
    
    llama_server_destroy(server);
    return 0;
}

int test_basic_inference() {
    void* server = NULL;
    llama_error_t err;
    
    // Setup
    err = llama_server_init(&server, server_config);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Server initialization");
    
    err = llama_server_load_model(server, test_model_path, NULL);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Model loading");
    
    int session_id;
    err = llama_server_create_session(server, session_config, &session_id);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Session creation");
    
    // Test completion inference
    const char* completion_input = R"({
        "prompt": "Hello, how are you?",
        "n_predict": 50,
        "temperature": 0.7
    })";
    
    int task_id;
    err = llama_server_create_task(server, session_id, "completion", completion_input, &task_id);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Task creation");
    
    char result[2048];
    size_t result_size;
    err = llama_server_get_result(server, task_id, result, sizeof(result), &result_size, 30000);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Get inference result");
    
    printf("Inference result: %s\n", result);
    TEST_ASSERT(strlen(result) > 0, "Result is not empty");
    
    // Cleanup
    llama_server_close_session(server, session_id);
    llama_server_destroy(server);
    return 0;
}

// ================================================================
// Concurrency Tests
// ================================================================

typedef struct {
    void* server;
    int session_id;
    int thread_id;
    int success;
} thread_test_data_t;

void* concurrent_inference_thread(void* arg) {
    thread_test_data_t* data = (thread_test_data_t*)arg;
    
    char input[512];
    snprintf(input, sizeof(input), R"({
        "prompt": "Hello from thread %d, tell me about AI",
        "n_predict": 30,
        "temperature": 0.7
    })", data->thread_id);
    
    int task_id;
    llama_error_t err = llama_server_create_task(data->server, data->session_id, 
                                                "completion", input, &task_id);
    if (err != LLAMA_SUCCESS) {
        printf("Thread %d: Failed to create task: %d\n", data->thread_id, err);
        data->success = 0;
        return NULL;
    }
    
    char result[1024];
    size_t result_size;
    err = llama_server_get_result(data->server, task_id, result, sizeof(result), 
                                 &result_size, 60000);
    if (err != LLAMA_SUCCESS) {
        printf("Thread %d: Failed to get result: %d\n", data->thread_id, err);
        data->success = 0;
        return NULL;
    }
    
    printf("Thread %d result: %.100s...\n", data->thread_id, result);
    data->success = 1;
    return NULL;
}

int test_concurrent_inference() {
    void* server = NULL;
    llama_error_t err;
    
    // Setup server
    err = llama_server_init(&server, server_config);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Server initialization");
    
    err = llama_server_load_model(server, test_model_path, NULL);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Model loading");
    
    // Create multiple sessions and threads
    const int num_threads = 3;
    pthread_t threads[num_threads];
    thread_test_data_t thread_data[num_threads];
    
    // Create sessions
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].server = server;
        thread_data[i].thread_id = i;
        thread_data[i].success = 0;
        
        err = llama_server_create_session(server, session_config, &thread_data[i].session_id);
        TEST_ASSERT(err == LLAMA_SUCCESS, "Session creation for concurrent test");
    }
    
    // Start concurrent inference
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, concurrent_inference_thread, &thread_data[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        TEST_ASSERT(thread_data[i].success == 1, "Concurrent inference thread success");
    }
    
    // Cleanup
    for (int i = 0; i < num_threads; i++) {
        llama_server_close_session(server, thread_data[i].session_id);
    }
    
    llama_server_destroy(server);
    return 0;
}

// ================================================================
// Streaming Tests
// ================================================================

typedef struct {
    char accumulated_text[2048];
    size_t total_length;
    int chunk_count;
    int final_received;
} stream_test_data_t;

int stream_callback(const char* chunk, size_t chunk_size, int is_final, void* user_data) {
    stream_test_data_t* data = (stream_test_data_t*)user_data;
    
    printf("Stream chunk %d: %.*s%s\n", data->chunk_count, (int)chunk_size, chunk, is_final ? " [FINAL]" : "");
    
    // Accumulate text
    size_t remaining = sizeof(data->accumulated_text) - data->total_length - 1;
    size_t to_copy = chunk_size < remaining ? chunk_size : remaining;
    
    if (to_copy > 0) {
        memcpy(data->accumulated_text + data->total_length, chunk, to_copy);
        data->total_length += to_copy;
        data->accumulated_text[data->total_length] = '\0';
    }
    
    data->chunk_count++;
    if (is_final) {
        data->final_received = 1;
    }
    
    return 0; // Continue streaming
}

int test_streaming_inference() {
    void* server = NULL;
    llama_error_t err;
    
    // Setup
    err = llama_server_init(&server, server_config);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Server initialization");
    
    err = llama_server_load_model(server, test_model_path, NULL);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Model loading");
    
    int session_id;
    err = llama_server_create_session(server, session_config, &session_id);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Session creation");
    
    // Test streaming
    const char* stream_input = R"({
        "prompt": "Tell me a short story about a robot",
        "n_predict": 100,
        "temperature": 0.7,
        "stream": true
    })";
    
    stream_test_data_t stream_data = {0};
    
    err = llama_server_stream_inference(server, session_id, stream_input, 
                                       stream_callback, &stream_data);
    TEST_ASSERT(err == LLAMA_SUCCESS, "Streaming inference");
    TEST_ASSERT(stream_data.chunk_count > 0, "Received stream chunks");
    TEST_ASSERT(stream_data.final_received == 1, "Received final chunk");
    TEST_ASSERT(strlen(stream_data.accumulated_text) > 0, "Accumulated text is not empty");
    
    printf("Final accumulated text: %s\n", stream_data.accumulated_text);
    
    // Cleanup
    llama_server_close_session(server, session_id);
    llama_server_destroy(server);
    return 0;
}

// ================================================================
// Utility Tests
// ================================================================

int test_error_handling() {
    // Test error messages
    const char* msg = llama_error_message(LLAMA_SUCCESS);
    TEST_ASSERT(strcmp(msg, "Success") == 0, "Success error message");
    
    msg = llama_error_message(LLAMA_ERROR_INVALID_ARGUMENT);
    TEST_ASSERT(strlen(msg) > 0, "Error message is not empty");
    
    // Test version
    const char* version = llama_server_version();
    TEST_ASSERT(strlen(version) > 0, "Version string is not empty");
    printf("Library version: %s\n", version);
    
    // Test invalid operations
    llama_error_t err = llama_server_destroy(NULL);
    TEST_ASSERT(err != LLAMA_SUCCESS, "Invalid server handle is rejected");
    
    return 0;
}

// ================================================================
// Main Test Runner
// ================================================================

int main(int argc, char** argv) {
    printf("=== Llama Server C API Test Suite ===\n");
    
    // Check if model file exists
    if (access(test_model_path, F_OK) != 0) {
        printf("WARNING: Test model not found at %s\n", test_model_path);
        printf("Please download a test model or update test_model_path\n");
        printf("Continuing with basic tests only...\n");
    }
    
    // Run tests
    TEST_FUNCTION(test_error_handling);
    
    // Only run server tests if model is available
    if (access(test_model_path, F_OK) == 0) {
        TEST_FUNCTION(test_server_lifecycle);
        TEST_FUNCTION(test_session_management);
        TEST_FUNCTION(test_basic_inference);
        TEST_FUNCTION(test_concurrent_inference);
        TEST_FUNCTION(test_streaming_inference);
    } else {
        printf("\nSkipping server tests due to missing model file\n");
    }
    
    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
