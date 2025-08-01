#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "../include/wasi_nn_llama.h"

// Function pointers for the APIs
typedef wasi_nn_error (*init_backend_func)(void **ctx);
typedef wasi_nn_error (*init_backend_with_config_func)(void **ctx, const char *config, uint32_t config_len);
typedef wasi_nn_error (*load_by_name_with_configuration_func)(void *ctx, const char *filename, uint32_t filename_len,
                                                              const char *config, uint32_t config_len, graph *g);
typedef wasi_nn_error (*init_execution_context_func)(void *ctx, graph g, graph_execution_context *exec_ctx);
typedef wasi_nn_error (*close_execution_context_func)(void *ctx, graph_execution_context exec_ctx);
typedef wasi_nn_error (*run_inference_func)(void *ctx, graph_execution_context exec_ctx, uint32_t index,
                                            tensor *input_tensor, tensor_data output_tensor, uint32_t *output_tensor_size);
typedef wasi_nn_error (*deinit_backend_func)(void *ctx);
int main()
{
    void *handle;
    init_backend_func init_backend;
    init_backend_with_config_func init_backend_with_config;
    load_by_name_with_configuration_func load_by_name_with_config;
    init_execution_context_func init_execution_context;
    close_execution_context_func close_execution_context;
    run_inference_func run_inference;
    deinit_backend_func deinit_backend;
    // Load the shared library
    handle = dlopen("./build/libwasi_nn_backend.so", RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        return EXIT_FAILURE;
    }
    printf("Library Open successfully.\n");
    // Clear any existing error
    dlerror();

    // Get function pointers
    *(void **)(&init_backend) = dlsym(handle, "init_backend");
    *(void **)(&init_backend_with_config) = dlsym(handle, "init_backend_with_config");
    *(void **)(&load_by_name_with_config) = dlsym(handle, "load_by_name_with_config");
    *(void **)(&init_execution_context) = dlsym(handle, "init_execution_context");
    *(void **)(&close_execution_context) = dlsym(handle, "close_execution_context");
    *(void **)(&run_inference) = dlsym(handle, "run_inference");
    *(void **)(&deinit_backend) = dlsym(handle, "deinit_backend");
    printf("Library Load successfully.\n");
    char *error;
    if ((error = dlerror()) != NULL)
    {
        fprintf(stderr, "%s\n", error);
        dlclose(handle);
        return EXIT_FAILURE;
    }

    void *backend_ctx = NULL;
    graph g = 0;
    graph_execution_context exec_ctx = 0;
    wasi_nn_error err;

    // Initialize backend with config
    const char *backend_config = "{"
                                 "\"max_sessions\":50,"
                                 "\"idle_timeout_ms\":600000,"
                                 "\"auto_cleanup\":true,"
                                 "\"max_concurrent\":4,"
                                 "\"queue_size\":20,"
                                 "\"memory_policy\":{"
                                 "\"context_shifting\":true,"
                                 "\"cache_strategy\":\"lru\","
                                 "\"max_cache_tokens\":5000"
                                 "},"
                                 "\"logging\":{"
                                 "\"level\":\"debug\","
                                 "\"enable_debug\":true,"
                                 "\"file\":\"/tmp/wasi_nn_backend.log\""
                                 "},"
                                 "\"performance\":{"
                                 "\"batch_processing\":true,"
                                 "\"batch_size\":256"
                                 "}"
                                 "}";
    err = init_backend_with_config(&backend_ctx, backend_config, strlen(backend_config));
    if (err != success)
    {
        fprintf(stderr, "Failed to initialize backend\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("Backend initialized successfully\n");

    // Load model with configuration
    const char *model_filename = "./test/qwen2.5-14b-instruct-q2_k.gguf"; // Update with your model file path
    const char *config = "{"
                         "\"n_gpu_layers\":98,"
                         "\"ctx_size\":2048,"
                         "\"n_predict\":512,"
                         "\"batch_size\":512,"
                         "\"threads\":8,"
                         "\"sampling\":{"
                         "\"temp\":0.7,"
                         "\"top_p\":0.95,"
                         "\"top_k\":40,"
                         "\"min_p\":0.05,"
                         "\"typical_p\":1.0,"
                         "\"repeat_penalty\":1.10,"
                         "\"presence_penalty\":0.0,"
                         "\"frequency_penalty\":0.0,"
                         "\"penalty_last_n\":64,"
                         "\"mirostat\":0,"
                         "\"mirostat_tau\":5.0,"
                         "\"mirostat_eta\":0.1,"
                         "\"seed\":-1"
                         "},"
                         "\"stopping\":{"
                         "\"max_tokens\":256,"
                         "\"max_time_ms\":30000,"
                         "\"ignore_eos\":false,"
                         "\"stop\":[\"\\n\\n\"]"
                         "},"
                         "\"memory\":{"
                         "\"context_shifting\":true"
                         "}"
                         "}";
    err = load_by_name_with_config(backend_ctx, model_filename, strlen(model_filename), config, strlen(config), &g);
    if (err != success)
    {
        fprintf(stderr, "Failed to load model\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("Model loaded successfully\n");

    // Initialize execution context
    err = init_execution_context(backend_ctx, g, &exec_ctx);
    if (err != success)
    {
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
    input_tensor.type = fp32;       // Assuming fp32 for this example

    // Prepare output tensor
    uint32_t output_tensor_size = 1024; // Adjust size as needed
    tensor_data output_tensor = (tensor_data)calloc(output_tensor_size, sizeof(uint8_t));
    if (output_tensor == NULL)
    {
        fprintf(stderr, "Failed to allocate output tensor\n");
        dlclose(handle);
        return EXIT_FAILURE;
    }

    // Run inference
    err = run_inference(backend_ctx, exec_ctx, 0, &input_tensor, output_tensor, &output_tensor_size);
    if (err != success)
    {
        fprintf(stderr, "Inference failed\n");
        free(output_tensor);
        dlclose(handle);
        return EXIT_FAILURE;
    }
    printf("\nrun Inference1 successful\n");
    printf("Output: %s\n", output_tensor);

    // Test concurrency limit and queue management
    printf("\n--- Testing concurrency limit and queue management ---\n");
    graph_execution_context exec_ctx2, exec_ctx3, exec_ctx4, exec_ctx5;

    // Try to create more sessions than the limit (max_concurrent=4)
    err = init_execution_context(backend_ctx, g, &exec_ctx2);
    if (err != success)
    {
        printf("Correctly rejected execution context 2 due to concurrency limit\n");
    }
    else
    {
        printf("Execution context 2 initialized successfully\n");

        err = init_execution_context(backend_ctx, g, &exec_ctx3);
        if (err != success)
        {
            printf("Correctly rejected execution context 3 due to concurrency limit\n");
        }
        else
        {
            printf("Execution context 3 initialized successfully\n");

            err = init_execution_context(backend_ctx, g, &exec_ctx4);
            if (err != success)
            {
                printf("Correctly rejected execution context 4 due to concurrency limit\n");
            }
            else
            {
                printf("Execution context 4 initialized successfully\n");

                // This should be queued because we've reached the concurrency limit
                // but there's still room in the queue (queue_size=20)
                err = init_execution_context(backend_ctx, g, &exec_ctx5);
                if (err != success)
                {
                    printf("Failed to initialize execution context 5\n");
                }
                else
                {
                    printf("Execution context 5 queued successfully\n");
                    
                    // Run inference on exec_ctx5 to test the queue
                    tensor input_tensor2;
                    const char *prompt2 = "What is the capital of France?";
                    input_tensor2.data = (tensor_data)prompt2;
                    input_tensor2.dimensions = NULL;
                    input_tensor2.type = fp32;
                    
                    uint32_t output_tensor_size2 = 1024;
                    tensor_data output_tensor2 = (tensor_data)calloc(output_tensor_size2, sizeof(uint8_t));
                    if (output_tensor2 == NULL)
                    {
                        fprintf(stderr, "Failed to allocate output tensor 2\n");
                    }
                    else
                    {
                        printf("Running inference on queued context...\n");
                        err = run_inference(backend_ctx, exec_ctx5, 0, &input_tensor2, output_tensor2, &output_tensor_size2);
                        if (err != success)
                        {
                            fprintf(stderr, "Queued inference failed\n");
                        }
                        else
                        {
                            printf("Queued inference successful\n");
                            printf("Output: %s\n", output_tensor2);
                        }
                        free(output_tensor2);
                    }
                    
                    // Close exec_ctx5 to clean up
                    close_execution_context(backend_ctx, exec_ctx5);
                }

                // Close one session and try again
                err = close_execution_context(backend_ctx, exec_ctx4);
                if (err != success)
                {
                    fprintf(stderr, "Failed to close execution context 4\n");
                }
                else
                {
                    printf("Execution context 4 closed successfully\n");
                }

                // Now this should succeed directly (not queued)
                graph_execution_context exec_ctx6;
                err = init_execution_context(backend_ctx, g, &exec_ctx6);
                if (err != success)
                {
                    fprintf(stderr, "Failed to initialize execution context 6\n");
                }
                else
                {
                    printf("Execution context 6 initialized successfully (not queued)\n");
                    // Close it to clean up
                    close_execution_context(backend_ctx, exec_ctx6);
                }
            }

            // Clean up remaining contexts
            close_execution_context(backend_ctx, exec_ctx3);
        }

        // Clean up remaining contexts
        close_execution_context(backend_ctx, exec_ctx2);
    }

    err = deinit_backend(backend_ctx);
    if (err != success)
    {
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
