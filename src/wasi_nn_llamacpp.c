/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "wasi_nn_types.h"
#include "utils/logger.h"
#include "stdlib.h"
#include "llama.h"
#include "ggml.h"
#include "cJSON.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// build info
extern int LLAMA_BUILD_NUMBER;
extern char const *LLAMA_COMMIT;
extern char const *LLAMA_COMPILER;
extern char const *LLAMA_BUILD_TARGET;

// compatable with WasmEdge
// https://github.com/second-state/WasmEdge-WASINN-examples/blob/master/wasmedge-ggml/README.md#parameters
// https://github.com/WasmEdge/WasmEdge/blob/master/plugins/wasi_nn/ggml.cpp
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
    llama_token *prompt;
    size_t prompt_len;
    llama_token *generation;
    size_t generation_len;
    struct wasi_nn_llama_config config;
};

struct EosDetector {
    // 配置参数
    char* eos_str;          // 要检测的EOS字符串
    int max_eos_len;        // EOS字符串最大长度
    
    // 环形缓冲区状态
    char* buffer;           // 字符缓冲区
    int buf_size;           // 缓冲区总容量
    int buf_pos;            // 当前写入位置
    int total_chars;        // 累计字符数（处理环形覆盖）
};

struct EosDetector* eos_detector_create(const char* eos_str) {
    struct EosDetector* d = malloc(sizeof(struct EosDetector));
    
    // 计算所需缓冲区大小（2倍EOS长度保证滑动窗口）
    int eos_len = strlen(eos_str);
    d->buf_size = eos_len * 2;
    d->buffer = calloc(d->buf_size, 1);
    
    // 初始化配置
    d->eos_str = strdup(eos_str);
    d->max_eos_len = eos_len;
    
    // 状态初始化
    d->buf_pos = 0;
    d->total_chars = 0;
    
    return d;
}

void eos_detector_free(struct EosDetector* d) {
    free(d->buffer);
    free(d->eos_str);
    free(d);
}

bool check_eos(struct EosDetector* d, const char* new_text) {
    // 将新文本写入环形缓冲区
    int len = strlen(new_text);
    for (int i = 0; i < len; i++) {
        d->buffer[d->buf_pos] = new_text[i];
        d->buf_pos = (d->buf_pos + 1) % d->buf_size;
        d->total_chars++;
    }
    
    // 计算需要检查的长度
    int check_len = MIN(d->max_eos_len, d->total_chars);
    if ((size_t)check_len < strlen(d->eos_str)) {
        return false;
    }
    
    // 构建最近check_len个字符的字符串
    char* recent = malloc(check_len + 1);
    for (int i = 0; i < check_len; i++) {
        int pos = (d->buf_pos - check_len + i + d->buf_size) % d->buf_size;
        recent[i] = d->buffer[pos];
    }
    recent[check_len] = '\0';
    
    // 检查是否包含EOS（使用strstr从末尾反向查找更高效）
    bool found = false;
    for (int i = check_len - strlen(d->eos_str); i >= 0; i--) {
        if (memcmp(recent + i, d->eos_str, strlen(d->eos_str)) == 0) {
            found = true;
            break;
        }
    }
    
    free(recent);
    return found;
}

// 全局初始化
struct EosDetector* g_eos_detector = NULL;
void init_eos_detector(const char *eos_str) {
    if (g_eos_detector) eos_detector_free(g_eos_detector);
    g_eos_detector = eos_detector_create(eos_str); // 默认不区分大小写
}

void auto_config_eos(struct LlamaContext* ctx) {
    llama_token eos_token = llama_token_eos(ctx->model);
    char buf[128] = { 0 };
    llama_token_to_piece(ctx->model, eos_token, buf, 120, 0, true);
    NN_INFO_PRINTF("Official EOS: %s\n", buf);
    init_eos_detector(buf);
}

static void
wasm_edge_llama_default_configuration(struct wasi_nn_llama_config *output)
{
    output->enable_log = true;
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

static void
llama_batch_clear(struct llama_batch *batch)
{
    batch->n_tokens = 0;
}

static void
llama_batch_add(struct llama_batch *batch, llama_token id, llama_pos pos,
                llama_seq_id *seq_ids, size_t seq_ids_len, bool logits)
{
    batch->token[batch->n_tokens] = id;
    batch->pos[batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids_len;
    for (size_t i = 0; i < seq_ids_len; ++i) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits[batch->n_tokens] = logits;

    batch->n_tokens++;
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

static void
llama_build_output_metadata(const struct LlamaContext *backend_ctx,
                            char *output_buf, size_t output_buf_size)
{
    snprintf(output_buf, output_buf_size,
             "{\"input_tokens\":%ld, \"output_tokens\":%ld, "
             "\"llama_build_number\":%d,"
             "\"llama_commit\":\"%s\"}",
             backend_ctx->prompt_len, backend_ctx->generation_len,
             LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
}

__attribute__((visibility("default"))) wasi_nn_error
init_backend(void **ctx)
{
    struct LlamaContext *backend_ctx = calloc(1, sizeof(struct LlamaContext));
    if (!backend_ctx) {
        NN_ERR_PRINTF("Allocate for OpenVINOContext failed");
        return runtime_error;
    }

    llama_backend_init();
    // llama_numa_init();
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
        llama_free(backend_ctx->ctx);

    if (backend_ctx->model)
        llama_free_model(backend_ctx->model);

    llama_backend_free();
    eos_detector_free(g_eos_detector);

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
        llama_load_model_from_file(filename, model_params);
    if (model == NULL) {
        NN_ERR_PRINTF("Failed to load model from file %s", filename);
        return runtime_error;
    }

    char buf[128] = { 0 };
    llama_model_desc(model, buf, 127);
    NN_INFO_PRINTF("Model desc %s", buf);

    backend_ctx->model = model;

    auto_config_eos(backend_ctx);

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
        llama_new_context_with_model(backend_ctx->model, ctx_params);
    if (llama_ctx == NULL) {
        NN_ERR_PRINTF("Failed to create context for model");
        return runtime_error;
    }

    backend_ctx->ctx = llama_ctx;

    NN_INFO_PRINTF("n_predict = %d, n_ctx = %d", backend_ctx->config.n_predict,
                   llama_n_ctx(backend_ctx->ctx));
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
set_input(void *ctx, graph_execution_context exec_ctx, uint32_t index,
          tensor *wasi_nn_tensor)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;
    // tensor->data is the prompt string. ends with \0
    char *prompt_text = (char *)wasi_nn_tensor->data;

    NN_DBG_PRINTF("prompt_text: %s", prompt_text);

    // tokenize the prompt
    uint32_t n_token_max = llama_n_ctx(backend_ctx->ctx);
    uint32_t prompt_text_len = strlen(prompt_text);

    if (backend_ctx->prompt == NULL) {
        backend_ctx->prompt = calloc(n_token_max, sizeof(llama_token));
        if (backend_ctx->prompt == NULL) {
            NN_ERR_PRINTF("Failed to allocate tokens_list");
            return runtime_error;
        }
    }

    int32_t n_tokens =
        llama_tokenize(backend_ctx->model, prompt_text, prompt_text_len,
                       backend_ctx->prompt, n_token_max, true, false);
    if (n_tokens < 0) {
        NN_ERR_PRINTF("Failed to tokenize prompt text");
        return runtime_error;
    }

    backend_ctx->prompt_len = n_tokens;

    NN_DBG_PRINTF("n_token_max: %d", n_token_max);
    NN_DBG_PRINTF("n_tokens: %d", n_tokens);

    // make sure the KV cache is big enough to hold all the prompt and generated
    // tokens
    int n_kv_req = n_tokens + (backend_ctx->config.n_predict - n_tokens);
    if (n_kv_req < 0 || (uint32_t)n_kv_req > n_token_max) {
        NN_ERR_PRINTF("the required KV cache size is not big enough, either "
                      "reduce n_predict or increase n_ctx");
        return runtime_error;
    }

    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
compute(void *ctx, graph_execution_context exec_ctx)
{
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;
    wasi_nn_error ret = runtime_error;

    // reset the generation buffer
    if (backend_ctx->generation == NULL) {
        backend_ctx->generation =
            calloc(backend_ctx->config.n_predict, sizeof(llama_token));
        if (backend_ctx->generation == NULL) {
            NN_ERR_PRINTF("Failed to allocate generation");
            return runtime_error;
        }
    }

    backend_ctx->generation_len = 0;

    // check KV cache
    uint32_t n_ctx = llama_n_ctx(backend_ctx->ctx);
    if (n_ctx <= backend_ctx->generation_len) {
        NN_ERR_PRINTF(
            "ctx_size(%u) is not big enough(<%ld), please increase it", n_ctx,
            backend_ctx->generation_len);
        return context_full;
    }

    // prepare the batch
    struct llama_batch batch =
        llama_batch_init(backend_ctx->config.batch_size, 0, 1);

    // evaluate the initial prompt
    llama_seq_id seq_ids[1] = { 0 };
    for (size_t i = 0; i < backend_ctx->prompt_len; i++) {
        llama_batch_add(&batch, backend_ctx->prompt[i], i, seq_ids,
                        sizeof(seq_ids) / sizeof(seq_ids[0]), false);
    }

    batch.logits[batch.n_tokens - 1] = true;

    if (batch.n_tokens > backend_ctx->config.n_predict) {
        NN_DBG_PRINTF("n_predict(%d) is not big enough(%d), please increase it",
                      backend_ctx->config.n_predict, batch.n_tokens);
        return prompt_tool_long;
    }

    if (llama_decode(backend_ctx->ctx, batch) != 0) {
        NN_ERR_PRINTF("First decode failed");
        return runtime_error;
    }

    // main loop
    int32_t n_cur = batch.n_tokens;
    int n_decode = 0;
    int32_t n_vocab = llama_n_vocab(backend_ctx->model);
    llama_token_data *candidates = NULL;

    candidates = calloc(n_vocab, sizeof(llama_token_data));
    if (candidates == NULL) {
        NN_ERR_PRINTF("Failed to allocate candidates");
        goto fail;
    }

    NN_DBG_PRINTF("n_cur: %d", n_cur);
    NN_DBG_PRINTF("n_predict: %d", backend_ctx->config.n_predict);

    while (n_cur <= backend_ctx->config.n_predict) {
        // sample the next token
        float *logits =
            llama_get_logits_ith(backend_ctx->ctx, batch.n_tokens - 1);

        memset(candidates, 0, sizeof(llama_token_data) * n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates[token_id].id = token_id;
            candidates[token_id].logit = logits[token_id];
            candidates[token_id].p = 0.0f;
        }

        llama_token_data_array candidates_p = { candidates, n_vocab, false };

        // sample the most likely token
        llama_token new_token_id =
            llama_sample_token_greedy(backend_ctx->ctx, &candidates_p);

        backend_ctx->generation[backend_ctx->generation_len++] = new_token_id;

        char buf[128] = { 0 };
        llama_token_to_piece(backend_ctx->model, new_token_id, buf, 120, 0,
                                true);
        // printf("%d(%s),", new_token_id, buf);

        // is it an end of generation?
        if (llama_token_is_eog(backend_ctx->model, new_token_id)) {
            printf("\n");
            NN_INFO_PRINTF("reach the end of generation");
            break;
        }
        // is it split to a seqential tokens?
        if (check_eos(g_eos_detector, buf)) {
            printf("\n");
            NN_INFO_PRINTF("reach the end of generation");
            break;
        }

        // prepare the next batch
        llama_batch_clear(&batch);
        // push this new token for next evaluation
        llama_batch_add(&batch, new_token_id, n_cur, seq_ids,
                        sizeof(seq_ids) / sizeof(seq_ids[0]), true);
        n_decode++;
        n_cur++;

        if (llama_decode(backend_ctx->ctx, batch) != 0) {
            NN_ERR_PRINTF("Secondary decode failed");
            goto fail;
        }
    }

    printf("\n");
    ret = success;
fail:
    llama_batch_free(batch);
    if (candidates != NULL) {
        free(candidates);
    }
    return ret;
}

__attribute__((visibility("default"))) wasi_nn_error
get_output(void *ctx, graph_execution_context exec_ctx, uint32_t index,
           tensor_data output_tensor, uint32_t *output_tensor_size)
{
    NN_DBG_PRINTF("output_tensor_size: %d", *output_tensor_size);
    struct LlamaContext *backend_ctx = (struct LlamaContext *)ctx;

    // Compatibility with WasmEdge
    if (index > 1) {
        NN_ERR_PRINTF("Invalid output index %d", index);
        return invalid_argument;
    }

    // Index 1 is for the metadata of the outputs.
    if (index == 1) {
        char output_metadata[128] = { 0 };
        llama_build_output_metadata(backend_ctx, output_metadata, 127);

        if (backend_ctx->config.stream_stdout) {
            printf("%s\n", output_metadata);
        }

        memcpy(output_tensor, output_metadata, strlen(output_metadata));
        *output_tensor_size = strlen(output_metadata);
        return success;
    }

    // token -> piece -> output_tensor
    if (backend_ctx->config.stream_stdout) {
        printf("\n");
    }

    NN_DBG_PRINTF("generation_len: %ld", backend_ctx->generation_len);

    size_t end_pos = 0;
    for (size_t i = 0; i < backend_ctx->generation_len; i++) {
        char buf[128] = { 0 };
        llama_token_to_piece(backend_ctx->model, backend_ctx->generation[i],
                             buf, 120, 0, true);

        if (backend_ctx->config.stream_stdout) {
            printf("%s", buf);
        }

        memcpy(output_tensor + end_pos, buf, strlen(buf));
        end_pos += strlen(buf);
    }

    if (backend_ctx->config.stream_stdout) {
        printf("\n");
    }
    
    NN_DBG_PRINTF("end_pos: %ld", end_pos);
    *output_tensor_size = end_pos;
    return success;
}

