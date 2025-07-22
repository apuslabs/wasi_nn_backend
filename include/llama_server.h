/*
 * Llama.cpp Server C API
 * 
 * A high-performance C library wrapper around llama.cpp server
 * providing concurrent inference capabilities through a simple C interface.
 */

#ifndef LLAMA_SERVER_H
#define LLAMA_SERVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error codes
 */
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

/**
 * Stream callback function type
 * @param chunk [in] Current data chunk
 * @param chunk_size [in] Size of the data chunk
 * @param is_final [in] Whether this is the final chunk
 * @param user_data [in] User data pointer
 * @return 0 to continue, non-zero to stop
 */
typedef int (*llama_stream_callback_t)(const char* chunk, size_t chunk_size, 
                                      int is_final, void* user_data);

/* ================================================================
 * Server Management APIs
 * ================================================================ */

/**
 * Initialize llama server instance
 * @param server_handle [out] Server handle
 * @param config_json [in] JSON configuration string, can be NULL for defaults
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_init(void** server_handle, const char* config_json);

/**
 * Load model into server
 * @param server_handle [in] Server handle
 * @param model_path [in] Path to model file
 * @param model_config_json [in] Model configuration JSON, can be NULL
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_load_model(void* server_handle, 
                                     const char* model_path,
                                     const char* model_config_json);

/**
 * Destroy server instance and free all resources
 * @param server_handle [in] Server handle
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_destroy(void* server_handle);

/* ================================================================
 * Session Management APIs
 * ================================================================ */

/**
 * Create a new inference session (maps to llama.cpp slot)
 * @param server_handle [in] Server handle
 * @param session_config [in] Session configuration JSON, can be NULL
 * @param session_id [out] Session ID for this user/conversation
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_create_session(void* server_handle,
                                         const char* session_config,
                                         int* session_id);

/**
 * Close and cleanup a session
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_close_session(void* server_handle, int session_id);

/* ================================================================
 * Inference APIs
 * ================================================================ */

/**
 * Create inference task within a session
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID (for conversation context)
 * @param task_type [in] Task type: "completion", "embedding", "rerank", "chat"
 * @param input_json [in] Input JSON string
 * @param task_id [out] Task ID
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_create_task(void* server_handle,
                                      int session_id,
                                      const char* task_type,
                                      const char* input_json,
                                      int* task_id);

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
__attribute__((visibility("default"))) 
llama_error_t llama_server_get_result(void* server_handle,
                                     int task_id,
                                     char* result_buffer,
                                     size_t buffer_size,
                                     size_t* actual_size,
                                     int timeout_ms);

/**
 * Stream inference within a session
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID
 * @param input_json [in] Input JSON string
 * @param callback [in] Stream callback function
 * @param user_data [in] User data pointer
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_stream_inference(void* server_handle,
                                           int session_id,
                                           const char* input_json,
                                           llama_stream_callback_t callback,
                                           void* user_data);

/**
 * Cancel running task
 * @param server_handle [in] Server handle
 * @param task_id [in] Task ID
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_cancel_task(void* server_handle, int task_id);

/* ================================================================
 * Status Query APIs
 * ================================================================ */

/**
 * Get server status
 * @param server_handle [in] Server handle
 * @param status_buffer [out] Status JSON buffer
 * @param buffer_size [in] Buffer size
 * @param actual_size [out] Actual status size
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_get_status(void* server_handle,
                                     char* status_buffer,
                                     size_t buffer_size,
                                     size_t* actual_size);

/**
 * Get session information and statistics
 * @param server_handle [in] Server handle
 * @param session_id [in] Session ID
 * @param info_buffer [out] Session info JSON buffer
 * @param buffer_size [in] Buffer size
 * @param actual_size [out] Actual info size
 * @return Error code
 */
__attribute__((visibility("default"))) 
llama_error_t llama_server_get_session_info(void* server_handle,
                                           int session_id,
                                           char* info_buffer,
                                           size_t buffer_size,
                                           size_t* actual_size);

/* ================================================================
 * Utility APIs
 * ================================================================ */

/**
 * Get error message string
 * @param error_code [in] Error code
 * @return Error message string (const, no need to free)
 */
__attribute__((visibility("default"))) 
const char* llama_error_message(llama_error_t error_code);

/**
 * Get library version
 * @return Version string (const, no need to free)
 */
__attribute__((visibility("default"))) 
const char* llama_server_version(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SERVER_H
