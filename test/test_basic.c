#include "test_common.h"

// Test 1: Basic Backend Initialization
int test_basic_backend_init() {
    void *backend_ctx = NULL;
    wasi_nn_error err;

    err = wasi_init_backend(&backend_ctx);
    ASSERT_SUCCESS(err, "Basic backend initialization failed");
    ASSERT(backend_ctx != NULL, "Backend context is NULL");

    err = wasi_deinit_backend(backend_ctx);
    ASSERT_SUCCESS(err, "Backend deinitialization failed");

    return 1;
}

// Test 2: Legacy Flat Configuration
int test_legacy_flat_config() {
    void *backend_ctx = NULL;
    wasi_nn_error err;

    const char *legacy_config = "{"
                               "\"max_sessions\":25,"
                               "\"idle_timeout_ms\":150000,"
                               "\"auto_cleanup\":false,"
                               "\"max_concurrent\":2,"
                               "\"queue_size\":10"
                               "}";

    err = wasi_init_backend_with_config(&backend_ctx, legacy_config, strlen(legacy_config));
    ASSERT_SUCCESS(err, "Legacy flat configuration failed");

    err = wasi_deinit_backend(backend_ctx);
    ASSERT_SUCCESS(err, "Backend cleanup failed");

    printf("✅ Legacy flat configuration working correctly\n");
    return 1;
}

// Test 3: Enhanced Nested Configuration
int test_enhanced_nested_config() {
    void *backend_ctx = NULL;
    wasi_nn_error err;

    const char *nested_config = "{"
                               "\"backend\":{"
                               "\"max_sessions\":100,"
                               "\"idle_timeout_ms\":300000,"
                               "\"auto_cleanup\":true,"
                               "\"max_concurrent\":8,"
                               "\"queue_size\":50"
                               "},"
                               "\"memory_policy\":{"
                               "\"context_shifting\":true,"
                               "\"cache_strategy\":\"lru\","
                               "\"max_cache_tokens\":10000"
                               "},"
                               "\"logging\":{"
                               "\"level\":\"info\","
                               "\"enable_debug\":false"
                               "},"
                               "\"performance\":{"
                               "\"batch_processing\":true,"
                               "\"batch_size\":512"
                               "}"
                               "}";

    err = wasi_init_backend_with_config(&backend_ctx, nested_config, strlen(nested_config));
    ASSERT_SUCCESS(err, "Enhanced nested configuration failed");

    err = wasi_deinit_backend(backend_ctx);
    ASSERT_SUCCESS(err, "Backend cleanup failed");

    printf("✅ Enhanced nested configuration working correctly\n");
    return 1;
}

// Test 4: Legacy Model Configuration
int test_legacy_model_config() {
    void *backend_ctx = NULL;
    graph g = 0;
    wasi_nn_error err;

    // Initialize backend first
    err = wasi_init_backend(&backend_ctx);
    ASSERT_SUCCESS(err, "Backend initialization failed");

    const char *legacy_model_config = "{"
                                     "\"n_gpu_layers\":48,"
                                     "\"ctx_size\":1024,"
                                     "\"n_predict\":256,"
                                     "\"batch_size\":256,"
                                     "\"threads\":4,"
                                     "\"temp\":0.8,"
                                     "\"top_p\":0.9,"
                                     "\"repeat_penalty\":1.05"
                                     "}";

    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  legacy_model_config, strlen(legacy_model_config), &g);
    ASSERT_SUCCESS(err, "Legacy model configuration failed");

    err = wasi_deinit_backend(backend_ctx);
    ASSERT_SUCCESS(err, "Backend cleanup failed");

    printf("✅ Legacy model configuration working correctly\n");
    return 1;
}

// Test 5: Enhanced Model Configuration with GPU
int test_enhanced_model_config() {
    void *backend_ctx = NULL;
    graph g = 0;
    wasi_nn_error err;

    err = wasi_init_backend(&backend_ctx);
    ASSERT_SUCCESS(err, "Backend initialization failed");

    const char *enhanced_model_config = "{"
                                       "\"model\":{"
                                       "\"n_gpu_layers\":98,"
                                       "\"ctx_size\":2048,"
                                       "\"n_predict\":512,"
                                       "\"batch_size\":512,"
                                       "\"threads\":8"
                                       "},"
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
                                       "\"stop\":[\"\\n\\n\",\"User:\",\"Assistant:\"],"
                                       "\"max_tokens\":512,"
                                       "\"max_time_ms\":30000,"
                                       "\"ignore_eos\":false"
                                       "},"
                                       "\"memory\":{"
                                       "\"context_shifting\":true,"
                                       "\"cache_prompt\":true,"
                                       "\"max_cache_tokens\":10000"
                                       "}"
                                       "}";

    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  enhanced_model_config, strlen(enhanced_model_config), &g);
    ASSERT_SUCCESS(err, "Enhanced model configuration failed");

    err = wasi_deinit_backend(backend_ctx);
    ASSERT_SUCCESS(err, "Backend cleanup failed");

    printf("✅ Enhanced model configuration with GPU working correctly\n");
    return 1;
}
