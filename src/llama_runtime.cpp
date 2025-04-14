#include "llama_runtime.h" 
#include "llama.h"         
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream> 
#include "cJSON.h"

struct llama_config {

    bool stream_stdout;
    bool enable_debug_log;

    // Model parameters (need to reload the model if updated):
    int32_t n_gpu_layers = 99;
    int32_t main_gpu;

    // Context parameters (used by the llama context):
    int32_t n_predict             =  1024; // new tokens to predict
    int32_t n_ctx                 =  0; // context size
    int32_t n_batch               =  1024; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t  n_threads; 
    int32_t  n_threads_batch;
    // Sampling parameters (used by the llama sampling context).
    float penalty_repeat         = 1.5f;
};


struct LlamaStateInternal {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;
    const llama_vocab* vocab = nullptr;
    struct llama_config config;
};

// --- Helper Function: Copy string safely ---

static void copy_string_safe(char* dest, size_t dest_size, const std::string& src) {
    if (!dest || dest_size == 0) return;
    strncpy(dest, src.c_str(), dest_size - 1);
    dest[dest_size - 1] = '\0'; // Ensure null termination
}



// --- Apply configs 
static void
llama_apply_configuration(const char *config_json,
                                    struct llama_config *output)
{
    cJSON *root = cJSON_Parse(config_json);
    if (root == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            printf("Error before: %s\n", error_ptr);
        }
        else {
            printf("Failed to parse JSON");
        }
        return;
    }

    cJSON *item = NULL;
    printf("parse configs %s \n",config_json);
    item = cJSON_GetObjectItem(root, "enable_debug_log");
    if (item != NULL) {
        output->enable_debug_log = cJSON_IsTrue(item);
        printf("apply enable-debug-log %d \n", output->enable_debug_log);
        if(!output->enable_debug_log)
        {
            llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
                if (level >= GGML_LOG_LEVEL_ERROR) { // Only log errors
                    fprintf(stderr, "%s", text);
                }
            }, nullptr);
        }
    }

    item = cJSON_GetObjectItem(root, "stream-stdout");
    if (item != NULL) {
        output->stream_stdout = cJSON_IsTrue(item);
        printf("apply stream-stdout %d \n", output->stream_stdout);
    }

    item = cJSON_GetObjectItem(root, "n-gpu-layers");
    if (item != NULL) {
        output->n_gpu_layers = (int)cJSON_GetNumberValue(item);
        printf("apply n_gpu_layers %d \n", output->n_gpu_layers);
    }

    item = cJSON_GetObjectItem(root, "main-gpu");
    if (item != NULL) {
        output->main_gpu = (int32_t)cJSON_GetNumberValue(item);
        printf("apply main_gpu %d\n", output->main_gpu);
    }

    item = cJSON_GetObjectItem(root, "n-predict");
    if (item != NULL) {
        output->n_predict = (int32_t)cJSON_GetNumberValue(item);
        printf("apply n-predict %d \n", output->n_predict);
    }



    item = cJSON_GetObjectItem(root, "n_ctx");
    if (item != NULL) {
        output->n_ctx = (int32_t)cJSON_GetNumberValue(item);
        printf("apply n_ctx %d \n", output->n_ctx);
    }

    item = cJSON_GetObjectItem(root, "n_batch");
    if (item != NULL) {
        output->n_batch = (int32_t)cJSON_GetNumberValue(item);
        printf("apply n_ctx %d \n", output->n_batch);
    }
    item = cJSON_GetObjectItem(root, "n_threads");
    if (item != NULL) {
        output->n_threads = (int32_t)cJSON_GetNumberValue(item);
        printf("apply n_ctx %d \n", output->n_threads);
    }
    item = cJSON_GetObjectItem(root, "n_threads_batch");
    if (item != NULL) {
        output->n_threads_batch = (int32_t)cJSON_GetNumberValue(item);
        printf("apply n_ctx %d \n", output->n_threads_batch);
    }



    item = cJSON_GetObjectItem(root, "penalty-repeat");
    if (item != NULL) {
        output->penalty_repeat = (float)cJSON_GetNumberValue(item);
        printf("apply penalty_repeat %f \n", output->penalty_repeat);
    }



    // more ...

    cJSON_Delete(root);
}


// --- API Implementation ---

LLAMA_RUNTIME_API LlamaHandle initialize_llama_runtime(
    const char* model_path,
    const char* config_json,
    char* error_msg_buffer,
    size_t error_msg_buffer_size
) {
    LlamaStateInternal* state = nullptr;
    try {
        state = new LlamaStateInternal(); // Allocate internal state
        llama_apply_configuration(config_json, &state->config);
        
        ggml_backend_load_all();

        // Load model
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = state->config.n_gpu_layers;
        state->model = llama_model_load_from_file(model_path, model_params);
        if (!state->model) {
            throw std::runtime_error("Unable to load model '" + std::string(model_path) + "'");
        }
        state->vocab = llama_model_get_vocab(state->model);

        // Create context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = state->config.n_ctx;
        ctx_params.n_batch = state->config.n_ctx; // Adjust batch size if needed
        ctx_params.n_threads = 8;
        ctx_params.n_threads_batch = 8;
        
        state->ctx = llama_init_from_model(state->model, ctx_params);
        if (!state->ctx) {
            throw std::runtime_error("Failed to create the llama_context");
        }

        // Initialize sampler (make parameters configurable if needed)
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);

       
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64,1.5,0.0,0.0));
        


        state->smpl = smpl;

        if (!state->smpl) {
            throw std::runtime_error("Failed to initialize sampler chain");
        }
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

LLAMA_RUNTIME_API bool run_inference(
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
    const char * tmpl = llama_model_chat_template(state->model, /* name */ nullptr);
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(state->ctx));
    int prev_len = 0;
    // add the user input to the message list and format it
    messages.push_back({"user", strdup(prompt_cstr)});
    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int)formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }
    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return 1;
    }

    // remove previous messages to obtain the prompt to generate the response
    std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
    
    
    printf("Prompt: %s\n", prompt.c_str());
    
    result_buffer[0] = '\0'; // Clear result buffer initially

    try {
        auto generate = [&](const std::string & prompt) {
            std::string response;
    
            const bool is_first = llama_kv_self_used_cells(state->ctx) == 0;
    
            // tokenize the prompt
            const int n_prompt_tokens = -llama_tokenize(state->vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
            std::vector<llama_token> prompt_tokens(n_prompt_tokens);
            if (llama_tokenize(state->vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
                GGML_ABORT("failed to tokenize the prompt\n");
            }
    
            // prepare a batch for the prompt
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
            llama_token new_token_id;
            while (true) {
                // check if we have enough space in the context to evaluate this batch
                int n_ctx = llama_n_ctx(state->ctx);
                int n_ctx_used = llama_kv_self_used_cells(state->ctx);
                if (n_ctx_used + batch.n_tokens > n_ctx) {
                    printf("\033[0m\n");
                    fprintf(stderr, "context size exceeded\n");
                    exit(0);
                }
    
                if (llama_decode(state->ctx, batch)) {
                    GGML_ABORT("failed to decode\n");
                }
    
                // sample the next token
                new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);
    
                // is it an end of generation?
                if (llama_vocab_is_eog(state->vocab, new_token_id)) {
                    break;
                }
    
                // convert the token to a string, print it and add it to the response
                char buf[256];
                int n = llama_token_to_piece(state->vocab, new_token_id, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    GGML_ABORT("failed to convert token to piece\n");
                }
                std::string piece(buf, n);
                printf("%s", piece.c_str());
                fflush(stdout);
                response += piece;
    
                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&new_token_id, 1);
            }
    
            return response;
        };
        std::string res = generate(prompt);
        copy_string_safe(result_buffer, result_buffer_size, res);
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
