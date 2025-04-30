#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "../include/wasi_nn_llama.h"

// Function pointers for the APIs
typedef wasi_nn_error (*init_backend_func)(void **ctx);
typedef wasi_nn_error (*load_by_name_with_configuration_func)(void *ctx, const char *filename, uint32_t filename_len,
                                            const char *config, uint32_t config_len, graph *g);
typedef wasi_nn_error (*init_execution_context_func)(void *ctx, graph g, graph_execution_context *exec_ctx);
typedef wasi_nn_error (*run_inference_func)(void *ctx, graph_execution_context exec_ctx, uint32_t index,
                                            tensor *input_tensor, tensor_data output_tensor, uint32_t *output_tensor_size);
typedef wasi_nn_error (*deinit_backend_func)(void *ctx);
int main() {
    void *handle;
    init_backend_func init_backend;
    load_by_name_with_configuration_func load_by_name_with_config;
    init_execution_context_func init_execution_context;
    run_inference_func run_inference;
    deinit_backend_func deinit_backend;
    // Load the shared library
    handle = dlopen("./build/libwasi_nn_backend.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        return EXIT_FAILURE;
    }
    printf("Library Open successfully.\n");
    // Clear any existing error
    dlerror();

    // Get function pointers
    *(void **) (&init_backend) = dlsym(handle, "init_backend");
    *(void **) (&load_by_name_with_config) = dlsym(handle, "load_by_name_with_config");
    *(void **) (&init_execution_context) = dlsym(handle, "init_execution_context");
    *(void **) (&run_inference) = dlsym(handle, "run_inference");
    *(void **) (&deinit_backend) = dlsym(handle, "deinit_backend");
    printf("Library Load successfully.\n");
    char *error;
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        dlclose(handle);
        return EXIT_FAILURE;
    }

    void *backend_ctx = NULL;
    graph g = 0;
    graph_execution_context exec_ctx = 0;
    wasi_nn_error err;

    // Initialize backend
    err = init_backend(&backend_ctx);
    if (err != success) {
        fprintf(stderr, "Failed to initialize backend\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("Backend initialized successfully\n");

    // Load model with configuration
    const char *model_filename = "./test/qwen1_5-14b-chat-q2_k.gguf"; // Update with your model file path
    const char* config = "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"stream-stdout\":true,\"enable_debug_log\":true}";
    err = load_by_name_with_config(backend_ctx, model_filename,strlen(model_filename),config,strlen(config), &g);
    if (err != success) {
        fprintf(stderr, "Failed to load model\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("Model loaded successfully\n");

    // Initialize execution context
    err = init_execution_context(backend_ctx, g, &exec_ctx);
    if (err != success) {
        fprintf(stderr, "Failed to initialize execution context\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("Execution context initialized successfully\n");

    // Prepare input tensor
    tensor input_tensor;
    const char *prompt1 = "Hello, I am Alex, who are you?";
    const char *prompt2 = "Do you know Arweave?";
    input_tensor.data = (tensor_data)prompt1;
    input_tensor.dimensions = NULL; // Assuming not needed for this example
    input_tensor.type = fp32; // Assuming fp32 for this example

    // Prepare output tensor
    uint32_t output_tensor_size = 1024; // Adjust size as needed
    tensor_data output_tensor = (tensor_data)calloc(output_tensor_size, sizeof(uint8_t));
    if (output_tensor == NULL) {
        fprintf(stderr, "Failed to allocate output tensor\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }

    // Run inference
    err = run_inference(backend_ctx, exec_ctx, 0, &input_tensor, output_tensor, &output_tensor_size);
    if (err != success) {
        fprintf(stderr, "Inference failed\n");
        free(output_tensor);
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("\nrun Inference1 successful\n");
    printf("Output: %s\n", output_tensor);
    input_tensor.data = (tensor_data)prompt2;
    err = run_inference(backend_ctx, exec_ctx, 0, &input_tensor, output_tensor, &output_tensor_size);
    if (err != success) {
        fprintf(stderr, "Inference failed\n");
        free(output_tensor);
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("\nrun Inference2 successful\n");
    // Print output
    printf("Output: %s\n", output_tensor);

    err = deinit_backend(backend_ctx);
    if (err != success) {
        fprintf(stderr, "Failed to Deinitialize backend\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("Backend deinitialized successfully\n");
    // Clean up
    free(output_tensor);
    dlclose(handle);

    return EXIT_SUCCESS;
}