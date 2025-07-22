/*
 * Simple example demonstrating llama_server C API usage
 */

#include "llama_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    // Configuration
    const char* model_path = (argc > 1) ? argv[1] : "./test/Qwen2.5-1.5B-Instruct.Q2_K.gguf";
    const char* server_config = R"({
        "n_parallel": 2,
        "n_ctx": 2048,
        "n_threads": 4
    })";
    const char* session_config = R"({
        "cache_prompt": true,
        "temperature": 0.7
    })";
    
    void* server = NULL;
    llama_error_t err;
    
    printf("=== Llama Server C API Example ===\n");
    
    // 1. Initialize server
    printf("Initializing server...\n");
    err = llama_server_init(&server, server_config);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to initialize server: %s\n", llama_error_message(err));
        return 1;
    }
    printf("Server initialized successfully.\n");
    
    // 2. Load model
    printf("Loading model from: %s\n", model_path);
    err = llama_server_load_model(server, model_path, NULL);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to load model: %s\n", llama_error_message(err));
        llama_server_destroy(server);
        return 1;
    }
    printf("Model loaded successfully.\n");
    
    // 3. Create session
    int session_id;
    printf("Creating session...\n");
    err = llama_server_create_session(server, session_config, &session_id);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to create session: %s\n", llama_error_message(err));
        llama_server_destroy(server);
        return 1;
    }
    printf("Session created with ID: %d\n", session_id);
    
    // 4. Run inference
    const char* input = R"({
        "prompt": "Hello! Can you tell me a short joke?",
        "n_predict": 100,
        "temperature": 0.7,
        "top_p": 0.95
    })";
    
    printf("\nRunning inference...\n");
    printf("Input prompt: Hello! Can you tell me a short joke?\n");
    
    int task_id;
    err = llama_server_create_task(server, session_id, "completion", input, &task_id);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to create task: %s\n", llama_error_message(err));
        goto cleanup;
    }
    printf("Task created with ID: %d\n", task_id);
    
    // 5. Get result
    char result[2048];
    size_t result_size;
    printf("Waiting for result...\n");
    
    err = llama_server_get_result(server, task_id, result, sizeof(result), 
                                 &result_size, 30000);
    if (err != LLAMA_SUCCESS) {
        printf("Failed to get result: %s\n", llama_error_message(err));
        goto cleanup;
    }
    
    printf("\n=== RESULT ===\n");
    printf("%s\n", result);
    printf("=== END RESULT ===\n");
    
    // 6. Show server status
    char status[1024];
    size_t status_size;
    err = llama_server_get_status(server, status, sizeof(status), &status_size);
    if (err == LLAMA_SUCCESS) {
        printf("\nServer status: %s\n", status);
    }
    
cleanup:
    // 7. Cleanup
    printf("\nCleaning up...\n");
    llama_server_close_session(server, session_id);
    llama_server_destroy(server);
    printf("Done.\n");
    
    return (err == LLAMA_SUCCESS) ? 0 : 1;
}
