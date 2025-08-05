#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <setjmp.h>
#include <time.h>

// Signal handler for segmentation faults
extern jmp_buf segfault_jmp;
extern volatile sig_atomic_t segfault_occurred;

void segfault_handler(int sig);

// Simple test framework
extern int test_count;
extern int test_passed;
extern int test_failed;

#define TEST_SECTION(name) \
    do { \
        printf("\n============================================================\n"); \
        printf("TEST SECTION: %s\n", name); \
        printf("============================================================\n"); \
    } while(0)

#define RUN_TEST(test_name, test_func) \
    do { \
        test_count++; \
        printf("\n[TEST %d] %s\n", test_count, test_name); \
        printf("----------------------------------------------------\n"); \
        if (test_func()) { \
            printf("✅ PASSED: %s\n", test_name); \
            test_passed++; \
        } else { \
            printf("❌ FAILED: %s\n", test_name); \
            test_failed++; \
        } \
    } while(0)

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("ASSERTION FAILED: %s\n", message); \
            return 0; \
        } \
    } while(0)

#define ASSERT_SUCCESS(err, message) \
    do { \
        if ((err) != 0) { \
            printf("ASSERTION FAILED: %s (error code: %d)\n", message, err); \
            return 0; \
        } \
    } while(0)

// Include WASI-NN types and errors
typedef enum {
    success = 0,
    invalid_argument = 1,
    invalid_encoding = 2,
    timeout = 3,
    runtime_error = 4,
    unsupported_operation = 5,
    too_large = 6,
    not_found = 7
} wasi_nn_error;

typedef uint32_t graph;
typedef uint32_t graph_execution_context;

typedef enum {
    fp16 = 0,
    fp32 = 1,
    fp64 = 2,
    bf16 = 3,
    u8 = 4,
    i32 = 5,
    i64 = 6
} tensor_type;

typedef struct {
    uint32_t *dimensions;
    uint32_t size;
} tensor_dimensions;

typedef struct {
    tensor_dimensions *dimensions;
    tensor_type type;
    uint8_t *data;
} tensor;

typedef uint8_t *tensor_data;

// Function pointers for the APIs
typedef wasi_nn_error (*init_backend_func_t)(void **ctx);
typedef wasi_nn_error (*init_backend_with_config_func_t)(void **ctx, const char *config, uint32_t config_len);
typedef wasi_nn_error (*load_by_name_with_configuration_func_t)(void *ctx, const char *filename, uint32_t filename_len,
                                                              const char *config, uint32_t config_len, graph *g);
typedef wasi_nn_error (*init_execution_context_func_t)(void *ctx, graph g, graph_execution_context *exec_ctx);
typedef wasi_nn_error (*init_execution_context_with_session_id_func_t)(void *ctx, const char *session_id, graph_execution_context *exec_ctx);
typedef wasi_nn_error (*close_execution_context_func_t)(void *ctx, graph_execution_context exec_ctx);
typedef wasi_nn_error (*run_inference_func_t)(void *ctx, graph_execution_context exec_ctx, uint32_t index,
                                            tensor *input_tensor, tensor_data output_tensor, uint32_t *output_tensor_size,
                                            const char *runtime_config, uint32_t config_len);
typedef wasi_nn_error (*set_input_func_t)(void *ctx, graph_execution_context exec_ctx, uint32_t index, tensor *input_tensor);
typedef wasi_nn_error (*compute_func_t)(void *ctx, graph_execution_context exec_ctx);
typedef wasi_nn_error (*get_output_func_t)(void *ctx, graph_execution_context exec_ctx, uint32_t index, 
                                          tensor_data output_tensor, uint32_t *output_tensor_size);
typedef wasi_nn_error (*deinit_backend_func_t)(void *ctx);

// Global function pointers
extern void *handle;
extern init_backend_func_t wasi_init_backend;
extern init_backend_with_config_func_t wasi_init_backend_with_config;
extern load_by_name_with_configuration_func_t wasi_load_by_name_with_config;
extern init_execution_context_func_t wasi_init_execution_context;
extern init_execution_context_with_session_id_func_t wasi_init_execution_context_with_session_id;
extern close_execution_context_func_t wasi_close_execution_context;
extern run_inference_func_t wasi_run_inference;
extern set_input_func_t wasi_set_input;
extern compute_func_t wasi_compute;
extern get_output_func_t wasi_get_output;
extern deinit_backend_func_t wasi_deinit_backend;

// Test configurations
extern const char *MODEL_FILE;
extern const char *MODEL_CONFIG;

// Global tensor dimensions for safe reuse
extern tensor_dimensions global_text_dims;

// Helper functions
void setup_tensor(tensor *t, const char *data);
int setup_library(void);

// Test function declarations
// Basic tests
int test_basic_backend_init(void);
int test_legacy_flat_config(void);
int test_enhanced_nested_config(void);
int test_legacy_model_config(void);
int test_enhanced_model_config(void);

// Inference tests
int test_basic_inference(void);
int test_advanced_sampling(void);
int test_dynamic_runtime_parameters(void);

// Session tests
int test_session_management(void);
int test_auto_session_cleanup(void);
int test_concurrency_management(void);

// Logging tests
int test_logging_configuration(void);
int test_advanced_logging_features(void);
int test_file_logging(void);

// Model tests
int test_safe_model_switch(void);

// Stopping tests
int test_advanced_stopping_criteria(void);
int test_grammar_based_stopping(void);
int test_dynamic_timeout_stopping(void);
int test_token_pattern_stopping(void);
int test_advanced_stopping_integration(void);

// Error tests
int test_error_handling(void);
int test_dangerous_edge_cases(void);

#endif // TEST_COMMON_H
