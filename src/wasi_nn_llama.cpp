/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "wasi_nn_types.h"
#include "utils/logger.h"
#include "stdlib.h"
#include "llama.h"
//#include "ggml.h"
#include "cJSON.h"
#include <vector>
#include <string>
#include <sstream> 
// build info
extern int LLAMA_BUILD_NUMBER;
extern char const *LLAMA_COMMIT;
extern char const *LLAMA_COMPILER;
extern char const *LLAMA_BUILD_TARGET;

struct wasi_nn_llama_config {
    // Backend(plugin in WasmEdge) parameters:
    bool enable_log;
    bool enable_debug_log;
    bool stream_stdout;
    // embedding mode
    bool embedding;
    // TODO: can it be -1?
    // can't bigger than ctx_size
    int32_t n_predict;
    char *reverse_prompt;

    // Used by LLaVA
    // multi-model project file
    char *mmproj;
    char *image;

    // Model parameters (need to reload the model if updated):
    // align to definition of struct llama_model_params
    int32_t n_gpu_layers;
    int32_t main_gpu;
    // limited size: llama_max_devices()
    float *tensor_split;
    bool use_mmap;

    // Context parameters (used by the llama context):
    uint32_t ctx_size;
    uint32_t batch_size;
    uint32_t ubatch_size;
    uint32_t threads;

    // Sampling parameters (used by the llama sampling context).
    float temp;
    float topP;
    float repeat_penalty;
    float presence_penalty;
    float frequency_penalty;
};

struct LlamaContext {
    struct llama_context *ctx;
    struct llama_model *model;
    struct llama_sampler * smpl;
    const llama_vocab *vocab;
    llama_token *prompt;
    size_t prompt_len;
    llama_token *generation;
    size_t generation_len;
    struct wasi_nn_llama_config config;
};


static void
wasm_edge_llama_default_configuration(struct wasi_nn_llama_config *output)
{
    output->enable_log = false;
    output->enable_debug_log = false;
    output->stream_stdout = true;
    output->embedding = false;
    output->n_predict = 512;
    output->reverse_prompt = NULL;

    output->mmproj = NULL;
    output->image = NULL;

    output->main_gpu = 0;
    output->n_gpu_layers = 0;
    output->tensor_split = NULL;
    output->use_mmap = true;

    // 0 = from model
    output->ctx_size = 0;
    output->batch_size = 512;
    output->ubatch_size = output->batch_size;
    output->threads = 1;

    output->temp = 0.7;
    output->topP = 0.95;
    output->repeat_penalty = 1.10;
    output->presence_penalty = 0.0;
    output->frequency_penalty = 0.0;
}


static void
wasm_edge_llama_apply_configuration(const char *config_json,
                                    struct wasi_nn_llama_config *output)
{
    cJSON *root = cJSON_Parse(config_json);
    if (root == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            NN_WARN_PRINTF("Error before: %s\n", error_ptr);
        }
        else {
            NN_WARN_PRINTF("Failed to parse JSON");
        }
        return;
    }

    cJSON *item = NULL;

    item = cJSON_GetObjectItem(root, "enable-log");
    if (item != NULL) {
        output->enable_log = cJSON_IsTrue(item);
        NN_DBG_PRINTF("apply enable-log %d", output->enable_log);
    }

    item = cJSON_GetObjectItem(root, "enable-debug-log");
    if (item != NULL) {
        output->enable_debug_log = cJSON_IsTrue(item);
        NN_DBG_PRINTF("apply enable-debug-log %d", output->enable_debug_log);
    }

    item = cJSON_GetObjectItem(root, "stream-stdout");
    if (item != NULL) {
        output->stream_stdout = cJSON_IsTrue(item);
        NN_DBG_PRINTF("apply stream-stdout %d", output->stream_stdout);
    }

    item = cJSON_GetObjectItem(root, "embedding");
    if (item != NULL) {
        output->embedding = cJSON_IsTrue(item);
        NN_DBG_PRINTF("apply embedding %d", output->embedding);
    }

    item = cJSON_GetObjectItem(root, "n-predict");
    if (item != NULL) {
        output->n_predict = (int32_t)cJSON_GetNumberValue(item);
        NN_DBG_PRINTF("apply n-predict %d", output->n_predict);
    }

    item = cJSON_GetObjectItem(root, "n-gpu-layers");
    if (item != NULL) {
        output->n_gpu_layers = (int32_t)cJSON_GetNumberValue(item);
        NN_DBG_PRINTF("apply n_gpu_layers %d", output->n_gpu_layers);
    }

    item = cJSON_GetObjectItem(root, "ctx-size");
    if (item != NULL) {
        output->ctx_size = (uint32_t)cJSON_GetNumberValue(item);
        NN_DBG_PRINTF("apply ctx-size %d", output->ctx_size);
    }

    // more ...

    cJSON_Delete(root);
}

static struct llama_model_params
llama_model_params_from_wasi_nn_llama_config(
    struct wasi_nn_llama_config *config)
{
    struct llama_model_params result = llama_model_default_params();

    // TODO: support more
    result.main_gpu = config->main_gpu;
    result.n_gpu_layers = config->n_gpu_layers;
    result.use_mmap = config->use_mmap;

    return result;
}


// Function to safely copy a string into tensor_data
void copy_string_to_tensor_data(tensor_data dest, uint32_t dest_size, const std::string &src) {
    if (dest == nullptr || dest_size == 0) {
        NN_ERR_PRINTF("Destination buffer is null or size is zero");
        return;
    }

    size_t src_size = src.size();
    if (src_size >= dest_size) {
        NN_WARN_PRINTF("Source string is too long, truncating");
        src_size = dest_size - 1; // Leave space for the null terminator
    }

    strncpy(reinterpret_cast<char*>(dest), src.c_str(), src_size);
    dest[src_size] = '\0'; // Ensure null termination
}
static struct llama_context_params
llama_context_params_from_wasi_nn_llama_config(
    struct wasi_nn_llama_config *config)
{
    struct llama_context_params result = llama_context_default_params();

    // TODO: support more
    result.n_ctx = config->ctx_size;
    // result.embeddings = config->embedding;

    return result;
}

// always output ERROR and WARN
// INFO needs enable_log
// DEBUG needs enable_debug_log
static void
llama_log_callback_local(enum ggml_log_level level, const char *text,
                         void *user_data)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)user_data;

    if (level == GGML_LOG_LEVEL_DEBUG && !backend_ctx->config.enable_debug_log)
        return;

    if (level == GGML_LOG_LEVEL_INFO && !backend_ctx->config.enable_log)
        return;

    printf("%s", text);
}

__attribute__((visibility("default"))) wasi_nn_error
init_backend(void **ctx)
{
    struct LlamaContext *backend_ctx = new LlamaContext();
    if (!backend_ctx) {
        NN_ERR_PRINTF("Allocate for Backend Context failed");
        return runtime_error;
    }
    ggml_backend_load_all();

    llama_log_set(llama_log_callback_local, backend_ctx);

    NN_INFO_PRINTF("llama_build_number: % d, llama_commit: %s, llama_compiler: "
                   "%s, llama_build_target: %s",
                   LLAMA_BUILD_NUMBER, LLAMA_COMMIT, LLAMA_COMPILER,
                   LLAMA_BUILD_TARGET);

    *ctx = (void *)backend_ctx;
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
deinit_backend(void *ctx)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;

    if (!backend_ctx)
        return invalid_argument;

    if (backend_ctx->generation)
        free(backend_ctx->generation);

    if (backend_ctx->prompt)
        free(backend_ctx->prompt);

    if (backend_ctx->ctx)
    {
        llama_free(backend_ctx->ctx);
        backend_ctx->ctx = nullptr;
    }
        
    if(backend_ctx->smpl)
    {
        llama_sampler_free(backend_ctx->smpl);
        backend_ctx->smpl = nullptr;        
    }
    if (backend_ctx->model)
    {
        llama_model_free(backend_ctx->model);
        backend_ctx->model = nullptr;
    }
        

    // ggml_backend_free();

    free(backend_ctx);
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
load(void *ctx, graph_builder_array *builder, graph_encoding encoding,
     execution_target target, graph *g)
{
    return unsupported_operation;
}

static wasi_nn_error
__load_by_name_with_configuration(void *ctx, const char *filename, graph *g)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;

    // make sure backend_ctx->config is initialized

    struct llama_model_params model_params =
        llama_model_params_from_wasi_nn_llama_config(&backend_ctx->config);
    struct llama_model *model =
        llama_model_load_from_file(filename, model_params);
    if (model == NULL) {
        NN_ERR_PRINTF("Failed to load model from file %s", filename);
        return runtime_error;
    }

    backend_ctx->model = model;
    backend_ctx->vocab = llama_model_get_vocab(model);

    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
load_by_name(void *ctx, const char *filename, uint32_t filename_len, graph *g)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;

    // use default params
    wasm_edge_llama_default_configuration(&backend_ctx->config);
    return __load_by_name_with_configuration(ctx, filename, g);
}

__attribute__((visibility("default"))) wasi_nn_error
load_by_name_with_config(void *ctx, const char *filename, uint32_t filename_len,
                         const char *config, uint32_t config_len, graph *g)
{
    NN_DBG_PRINTF("filename: %s", filename);
    NN_DBG_PRINTF("config: %s", config);
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;

    wasm_edge_llama_default_configuration(&backend_ctx->config);

    if (config != NULL) {
        // parse wasmedge config
        wasm_edge_llama_apply_configuration(config, &backend_ctx->config);
    }
    else {
        NN_INFO_PRINTF("No configuration provided, use default");
    }

    return __load_by_name_with_configuration(ctx, filename, g);
}

// It is assumed that model params shouldn't be changed in Config stage.
// We only load the model once in the Load stage.
__attribute__((visibility("default"))) wasi_nn_error
init_execution_context(void *ctx, graph g, graph_execution_context *exec_ctx)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;

    struct llama_context_params ctx_params =
        llama_context_params_from_wasi_nn_llama_config(&backend_ctx->config);
    struct llama_context *llama_ctx =
        llama_init_from_model(backend_ctx->model, ctx_params);
    if (llama_ctx == NULL) {
        NN_ERR_PRINTF("Failed to create context for model");
        return runtime_error;
    }
    // Initialize sampler (make parameters configurable if needed)
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64,1.5,0.0,0.0));
    
    if (smpl == NULL) {
        NN_ERR_PRINTF("Failed to create smpl for model");
        return runtime_error;
    }
    backend_ctx->ctx = llama_ctx;
    backend_ctx->smpl = smpl;

    NN_INFO_PRINTF("n_predict = %d, n_ctx = %d", backend_ctx->config.n_predict,
                   llama_n_ctx(backend_ctx->ctx));
    return success;
}
__attribute__((visibility("default"))) wasi_nn_error
run_inference(void *ctx, graph_execution_context exec_ctx, uint32_t index,
          tensor *input_tensor,tensor_data output_tensor, uint32_t *output_tensor_size)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;
    char *prompt_text = (char *)input_tensor->data;
    
    const char * tmpl = llama_model_chat_template(backend_ctx->model, /* name */ nullptr);
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(backend_ctx->ctx));
    int prev_len = 0;
    // add the user input to the message list and format it
    messages.push_back({"user", strdup(prompt_text)});
    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int)formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }
    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return runtime_error;
    }

    // remove previous messages to obtain the prompt to generate the response
    std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
    
    
    printf("Prompt: %s\n", prompt.c_str());
    


    try {
        auto generate = [&](const std::string & prompt) {
            std::string response;
    
            const bool is_first = llama_kv_self_used_cells(backend_ctx->ctx) == 0;
    
            // tokenize the prompt
            const int n_prompt_tokens = -llama_tokenize(backend_ctx->vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
            std::vector<llama_token> prompt_tokens(n_prompt_tokens);
            if (llama_tokenize(backend_ctx->vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
                GGML_ABORT("failed to tokenize the prompt\n");
            }
    
            // prepare a batch for the prompt
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
            llama_token new_token_id;
            while (true) {
                // check if we have enough space in the context to evaluate this batch
                int n_ctx = llama_n_ctx(backend_ctx->ctx);
                int n_ctx_used = llama_kv_self_used_cells(backend_ctx->ctx);
                if (n_ctx_used + batch.n_tokens > n_ctx) {
                    printf("\033[0m\n");
                    fprintf(stderr, "context size exceeded\n");
                    exit(0);
                }
    
                if (llama_decode(backend_ctx->ctx, batch)) {
                    GGML_ABORT("failed to decode\n");
                }
    
                // sample the next token
                new_token_id = llama_sampler_sample(backend_ctx->smpl, backend_ctx->ctx, -1);
    
                // is it an end of generation?
                if (llama_vocab_is_eog(backend_ctx->vocab, new_token_id)) {
                    break;
                }
    
                // convert the token to a string, print it and add it to the response
                char buf[256];
                int n = llama_token_to_piece(backend_ctx->vocab, new_token_id, buf, sizeof(buf), 0, true);
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
        copy_string_to_tensor_data(output_tensor, *output_tensor_size, res);
        return success; // Success

    } catch (const std::exception& e) {
        std::string error_str = "Inference failed: " + std::string(e.what());
        return runtime_error; // Indicate failure
    }
    return success;
}



__attribute__((visibility("default"))) wasi_nn_error
set_input(void *ctx, graph_execution_context exec_ctx, uint32_t index,
          tensor *wasi_nn_tensor)
{
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
compute(void *ctx, graph_execution_context exec_ctx)
{
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
get_output(void *ctx, graph_execution_context exec_ctx, uint32_t index,
           tensor_data output_tensor, uint32_t *output_tensor_size)
{
    return success;
}

