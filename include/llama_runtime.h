#ifndef LLAMA_RUNTIME_H
#define LLAMA_RUNTIME_H

#include <stddef.h> // For size_t
#include <stdbool.h> // For bool

// --- Opaque Handle ---
// The user interacts with this handle, implementation details are hidden.
// 用户与此句柄交互，实现细节被隐藏。
typedef struct LlamaStateInternal* LlamaHandle;

// --- API Visibility Macros ---
// Define import/export macros for cross-platform compatibility (simplified for Linux/macOS)
// 为跨平台兼容性定义导入/导出宏（针对 Linux/macOS 简化）
#ifdef _WIN32
    #ifdef LLAMA_RUNTIME_BUILD_SHARED // Building the DLL
        #define LLAMA_RUNTIME_API __declspec(dllexport)
    #else // Using the DLL
        #define LLAMA_RUNTIME_API __declspec(dllimport)
    #endif
#else // Non-Windows (Linux, macOS)
    #define LLAMA_RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化 LLaMA 运行时环境。
 * Initializes the LLaMA runtime environment.
 *
 * @param model_path 模型文件的路径 (Path to the model file).
 * @param n_gpu_layers 要卸载到 GPU 的层数 (Number of layers to offload to GPU).
 * @param n_ctx 上下文大小 (Context size).
 * @param error_msg_buffer (输出参数) 用于存储错误信息的缓冲区 (Output buffer for error messages).
 * @param error_msg_buffer_size 错误信息缓冲区的大小 (Size of the error message buffer).
 * @return LlamaHandle 成功时返回有效的句柄，失败时返回 NULL (Returns a valid handle on success, NULL on failure).
 * 如果失败，错误信息会写入 error_msg_buffer。
 * (If it fails, an error message is written to error_msg_buffer).
 */
LLAMA_RUNTIME_API LlamaHandle initialize_llama_runtime(
    const char* model_path,
    int n_gpu_layers,
    int n_ctx,
    char* error_msg_buffer,
    size_t error_msg_buffer_size
);

/**
 * @brief 使用已初始化的 LLaMA 运行时执行推理。
 * Runs inference using the initialized LLaMA runtime.
 *
 * @param handle 从 initialize_llama_runtime 获取的有效句柄 (Valid handle obtained from initialize_llama_runtime).
 * @param prompt 输入的提示符 (The input prompt).
 * @param result_buffer (输出参数) 用于存储推理结果的缓冲区 (Output buffer for the inference result).
 * @param result_buffer_size 结果缓冲区的大小 (Size of the result buffer).
 * @param error_msg_buffer (输出参数) 用于存储错误信息的缓冲区 (Output buffer for error messages).
 * @param error_msg_buffer_size 错误信息缓冲区的大小 (Size of the error message buffer).
 * @return bool 成功时返回 true，失败时返回 false (Returns true on success, false on failure).
 * 如果失败，错误信息会写入 error_msg_buffer。
 * (If it fails, an error message is written to error_msg_buffer).
 */
LLAMA_RUNTIME_API bool run_llama_inference(
    LlamaHandle handle,
    const char* prompt,
    char* result_buffer,
    size_t result_buffer_size,
    char* error_msg_buffer,
    size_t error_msg_buffer_size
);

/**
 * @brief 清理并释放 LLaMA 运行时资源。
 * Cleans up and releases LLaMA runtime resources.
 *
 * @param handle 要释放的 LLaMA 句柄 (The LLaMA handle to release).
 */
LLAMA_RUNTIME_API void cleanup_llama_runtime(LlamaHandle handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LLAMA_RUNTIME_H
