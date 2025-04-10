#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>    // For dynamic loading functions
#include <stdbool.h>
#include <string.h>

// Include the public header for your library
// Adjust the path if necessary (e.g., "../include/llama_runtime.h")
#include "../include/llama_runtime.h"

// --- Configuration ---
// Path to the shared library (adjust if needed)
const char* LIB_PATH = "./build/libllama_runtime.so"; // Assumes it's in the current dir or build dir

// Model configuration (adjust to your model)
const char* MODEL_PATH = "./test/Qwen2.5-1.5B-Instruct.Q2_K.gguf"; // IMPORTANT: Set correct model path!
const int N_GPU_LAYERS = 99; // Or your desired number
const int N_CTX = 2048;      // Or your desired context size

// Buffer sizes
#define RESULT_BUFFER_SIZE 4096
#define ERROR_BUFFER_SIZE 256

int main() {
    void *lib_handle = NULL;
    LlamaHandle llama_handle = NULL;
    char result_buffer[RESULT_BUFFER_SIZE];
    char error_buffer[ERROR_BUFFER_SIZE];

    // --- Function Pointers ---
    // Declare pointers for the functions we want to load from the SO
    LlamaHandle (*initialize_func)(const char*, int, int, char*, size_t);
    bool (*inference_func)(LlamaHandle, const char*, char*, size_t, char*, size_t);
    void (*cleanup_func)(LlamaHandle);

    printf("Attempting to load library: %s\n", LIB_PATH);

    // --- 1. Load the Shared Library ---
    // RTLD_LAZY: Resolve symbols only when needed
    // RTLD_NOW: Resolve all symbols immediately (useful for checking)
    lib_handle = dlopen(LIB_PATH, RTLD_LAZY);
    if (!lib_handle) {
        fprintf(stderr, "Error loading library '%s': %s\n", LIB_PATH, dlerror());
        return 1;
    }
    printf("Library loaded successfully.\n");

    // Clear any previous errors
    dlerror();

    // --- 2. Get Pointers to Functions using dlsym ---
    *(void **) (&initialize_func) = dlsym(lib_handle, "initialize_llama_runtime");
    char *dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "Error getting symbol 'initialize_llama_runtime': %s\n", dlsym_error);
        dlclose(lib_handle);
        return 1;
    }

    *(void **) (&inference_func) = dlsym(lib_handle, "run_llama_inference");
    dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "Error getting symbol 'run_llama_inference': %s\n", dlsym_error);
        dlclose(lib_handle);
        return 1;
    }

    *(void **) (&cleanup_func) = dlsym(lib_handle, "cleanup_llama_runtime");
    dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "Error getting symbol 'cleanup_llama_runtime': %s\n", dlsym_error);
        dlclose(lib_handle);
        return 1;
    }
    printf("API functions loaded successfully.\n");

    // --- 3. Call Initialization Function ---
    printf("Initializing LLaMA runtime (Model: %s)...\n", MODEL_PATH);
    error_buffer[0] = '\0'; // Clear error buffer
    llama_handle = initialize_func(MODEL_PATH, N_GPU_LAYERS, N_CTX, error_buffer, ERROR_BUFFER_SIZE);

    if (!llama_handle) {
        fprintf(stderr, "Initialization failed: %s\n", error_buffer[0] ? error_buffer : "Unknown error");
        dlclose(lib_handle);
        return 1;
    }
    printf("LLaMA runtime initialized.\n");

    // --- 4. Call Inference Function ---
    const char* prompt = "Translate the following English text to French: 'Hello, world!'";
    printf("\nRunning inference with prompt: \"%s\"\n", prompt);
    result_buffer[0] = '\0'; // Clear result buffer
    error_buffer[0] = '\0';  // Clear error buffer

    bool success = inference_func(llama_handle, prompt, result_buffer, RESULT_BUFFER_SIZE, error_buffer, ERROR_BUFFER_SIZE);

    if (success) {
        printf("Inference successful.\n");
        printf("Result:\n---\n%s\n---\n", result_buffer);
        if (error_buffer[0] != '\0') {
             printf("Note: Inference reported a non-fatal issue: %s\n", error_buffer);
        }
    } else {
        fprintf(stderr, "Inference failed: %s\n", error_buffer[0] ? error_buffer : "Unknown error");
    }

    // --- 5. Call Cleanup Function ---
    printf("\nCleaning up LLaMA runtime...\n");
    cleanup_func(llama_handle);
    printf("Cleanup complete.\n");

    // --- 6. Unload the Library ---
    if (dlclose(lib_handle) != 0) {
         fprintf(stderr, "Error closing library: %s\n", dlerror());
         // Continue execution, but report error
    } else {
        printf("Library unloaded.\n");
    }


    return 0;
}
