#include "llama_runtime.h" // Include the public C header
#include "llama.h"         // Include the llama.cpp header

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream> // For formatting error messages

// --- Internal State Structure ---
// This struct is hidden from the user of the shared library.

struct LlamaStateInternal {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_ctx = 0;
};

// --- Helper Function: Copy string safely ---

static void copy_string_safe(char* dest, size_t dest_size, const std::string& src) {
    if (!dest || dest_size == 0) return;
    strncpy(dest, src.c_str(), dest_size - 1);
    dest[dest_size - 1] = '\0'; // Ensure null termination
}

// --- Helper Function: Token to Piece (Internal) ---

static std::string token_to_piece_internal(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        // In a real library, better error propagation might be needed
        // 在实际库中可能需要更好的错误传播
        fprintf(stderr, "Warning: Failed to convert token %d to piece.\n", token);
        return "";
    }
    return std::string(buf, n);
}


// --- API Implementation ---

LLAMA_RUNTIME_API LlamaHandle initialize_llama_runtime(
    const char* model_path,
    int n_gpu_layers,
    int n_ctx_param,
    char* error_msg_buffer,
    size_t error_msg_buffer_size
) {
    LlamaStateInternal* state = nullptr;
    try {
        state = new LlamaStateInternal(); // Allocate internal state
        state->n_ctx = n_ctx_param;

        // Set log callback (can be made configurable)
        llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) { // Only log errors
                fprintf(stderr, "%s", text);
            }
        }, nullptr);

        // Load backends (consider if this should be done once globally)
        ggml_backend_load_all();

        // Load model
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        state->model = llama_model_load_from_file(model_path, model_params);
        if (!state->model) {
            throw std::runtime_error("Unable to load model '" + std::string(model_path) + "'");
        }
        state->vocab = llama_model_get_vocab(state->model);

        // Create context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = state->n_ctx;
        ctx_params.n_batch = state->n_ctx; // Adjust batch size if needed
        state->ctx = llama_init_from_model(state->model, ctx_params);
        if (!state->ctx) {
            throw std::runtime_error("Failed to create the llama_context");
        }

        // Initialize sampler (make parameters configurable if needed)
        state->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
         if (!state->smpl) {
            throw std::runtime_error("Failed to initialize sampler chain");
        }
        llama_sampler_chain_add(state->smpl, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(state->smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(state->smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        printf("LLaMA runtime initialized successfully.\n");
        return static_cast<LlamaHandle>(state); // Return opaque handle

    } catch (const std::exception& e) {
        std::string error_str = "Initialization failed: " + std::string(e.what());
        copy_string_safe(error_msg_buffer, error_msg_buffer_size, error_str);
        // Cleanup partially initialized resources if error occurred
        if (state) {
            if (state->ctx) llama_free(state->ctx);
            if (state->model) llama_model_free(state->model);
             if (state->smpl) llama_sampler_free(state->smpl);
            delete state;
        }
        return NULL; // Return NULL on failure
    }
}

LLAMA_RUNTIME_API bool run_llama_inference(
    LlamaHandle handle,
    const char* prompt_cstr,
    char* result_buffer,
    size_t result_buffer_size,
    char* error_msg_buffer,
    size_t error_msg_buffer_size
) {
    if (!handle) {
        copy_string_safe(error_msg_buffer, error_msg_buffer_size, "Invalid handle passed to run_llama_inference.");
        return false;
    }
    // Cast opaque handle back to internal struct pointer
    LlamaStateInternal* state = static_cast<LlamaStateInternal*>(handle);

    if (!state->model || !state->ctx || !state->smpl || !state->vocab) {
         copy_string_safe(error_msg_buffer, error_msg_buffer_size, "Internal state is invalid.");
        return false;
    }
    if (!prompt_cstr) {
        copy_string_safe(error_msg_buffer, error_msg_buffer_size, "Prompt cannot be null.");
        return false;
    }
    if (!result_buffer || result_buffer_size == 0) {
        copy_string_safe(error_msg_buffer, error_msg_buffer_size, "Result buffer is invalid or has zero size.");
        return false;
    }

    std::string prompt(prompt_cstr);
    std::string response = "";
    result_buffer[0] = '\0'; // Clear result buffer initially

    try {
        // --- Reset state if needed (uncomment if each call should be independent) ---
        // llama_sampler_reset(state->smpl);
        // llama_kv_cache_clear(state->ctx);

        const bool is_first = true; // Assume start of new sequence

        // Tokenize prompt
        int n_prompt_tokens_estimated = -llama_tokenize(state->vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
        if (n_prompt_tokens_estimated >= state->n_ctx) {
            throw std::runtime_error("Prompt is too long for the context size.");
        }
        std::vector<llama_token> prompt_tokens(n_prompt_tokens_estimated);
        int n_prompt_tokens = llama_tokenize(state->vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true);
        if (n_prompt_tokens < 0) {
            throw std::runtime_error("Failed to tokenize the prompt.");
        }
        prompt_tokens.resize(n_prompt_tokens);

        // Prepare batch and decode/sample loop
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        int n_decoded = 0;

        // Decode initial prompt
        if (llama_kv_self_used_cells(state->ctx) + batch.n_tokens > state->n_ctx) {
             throw std::runtime_error("Context size exceeded by initial prompt (unexpected).");
        }
        if (llama_decode(state->ctx, batch)) {
            throw std::runtime_error("Failed to decode initial prompt.");
        }
        n_decoded += batch.n_tokens;

        // Generation loop
        size_t current_response_len = 0;
        while (true) {
            new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);

            if (llama_vocab_is_eog(state->vocab, new_token_id)) {
                break; // End of generation
            }

            std::string piece = token_to_piece_internal(state->vocab, new_token_id);
            if (current_response_len + piece.length() < result_buffer_size) {
                 memcpy(result_buffer + current_response_len, piece.c_str(), piece.length());
                 current_response_len += piece.length();
                 result_buffer[current_response_len] = '\0'; // Keep null-terminated
            } else {
                 fprintf(stderr, "\nWarning: Result buffer overflow during generation.\n");
                 copy_string_safe(error_msg_buffer, error_msg_buffer_size, "Result buffer overflow.");
                 // Return true as we got partial results, but indicate overflow via error msg
                 return true;
            }


            batch = llama_batch_get_one(&new_token_id, 1);

            if (llama_kv_self_used_cells(state->ctx) + batch.n_tokens > state->n_ctx) {
                fprintf(stderr, "\nWarning: Context size exceeded during generation.\n");
                // Return true as we got partial results
                return true;
            }

            if (llama_decode(state->ctx, batch)) {
                fprintf(stderr, "\nWarning: Failed to decode during generation.\n");
                 // Return true as we got partial results
                return true;
            }
            n_decoded++;
        }

        return true; // Success

    } catch (const std::exception& e) {
        std::string error_str = "Inference failed: " + std::string(e.what());
        copy_string_safe(error_msg_buffer, error_msg_buffer_size, error_str);
        return false; // Indicate failure
    }
}


LLAMA_RUNTIME_API void cleanup_llama_runtime(LlamaHandle handle) {
    if (!handle) {
        return;
    }
    LlamaStateInternal* state = static_cast<LlamaStateInternal*>(handle);

    if (state->smpl) {
        llama_sampler_free(state->smpl);
        state->smpl = nullptr;
    }
    if (state->ctx) {
        llama_free(state->ctx);
        state->ctx = nullptr;
    }
    if (state->model) {
        llama_model_free(state->model);
        state->model = nullptr;
    }
    // Consider ggml_backend_free() if appropriate for your application lifecycle
    // ggml_backend_free();

    delete state; // Free the internal state struct itself
    printf("LLaMA runtime resources cleaned up.\n");
}
