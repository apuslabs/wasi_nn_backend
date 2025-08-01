/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wasi_nn_llama.h"
#include "cJSON.h"
#include "utils/logger.h"

// Include llama.cpp headers
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

// Include all server types and structures directly from server.cpp
// We need the complete definitions, not just forward declarations
#include "server/server.cpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct SessionInfo
{
  std::string session_id;
  std::vector<common_chat_msg> chat_history;
  std::chrono::steady_clock::time_point last_activity;
};

struct LlamaChatContext
{
  // Server context (from server.cpp)
  server_context server_ctx;

  // Session management (updated)
  std::unordered_map<graph_execution_context, SessionInfo> sessions;
  graph_execution_context next_exec_ctx_id;

  // Auto-cleanup configuration
  uint32_t max_sessions;
  uint32_t idle_timeout_ms;
  bool auto_cleanup_enabled;

  // Concurrency and queue management
  uint32_t max_concurrent;
  uint32_t queue_size;
  uint32_t active_sessions; // Track active sessions

  // Memory policy
  bool context_shifting_enabled;
  std::string cache_strategy;
  uint32_t max_cache_tokens;

  // Logging configuration
  std::string log_level;
  bool enable_debug_log;
  std::string log_file;

  // Performance settings
  bool batch_processing_enabled;
  uint32_t batch_size;

  LlamaChatContext()
      : next_exec_ctx_id(1),
        max_sessions(100), idle_timeout_ms(300000), auto_cleanup_enabled(true),
        max_concurrent(8), queue_size(50), active_sessions(0),
        context_shifting_enabled(true), cache_strategy("lru"), max_cache_tokens(10000),
        log_level("info"), enable_debug_log(false),
        batch_processing_enabled(true), batch_size(512) {}
};

// Helper function to parse JSON config into common_params
static void parse_config_to_params(const char *config_json,
                                   common_params &params)
{
  // Set defaults (similar to main.cpp)
  params = common_params();
  params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
  params.enable_chat_template = true;
  params.n_predict = 512;
  params.sampling.temp = 0.7f;
  params.sampling.top_p = 0.95f;
  params.sampling.penalty_repeat = 1.10f;
  params.n_ctx = 2048;
  params.n_batch = 512;
  params.cpuparams.n_threads = 8;
  params.cpuparams_batch.n_threads = 8;
  
  // GPU acceleration defaults - explicitly set to enable GPU usage
  params.n_gpu_layers = 0;  // Will be overridden by config if specified

  if (!config_json)
    return;

  cJSON *root = cJSON_Parse(config_json);
  if (!root)
  {
    NN_WARN_PRINTF("Failed to parse config JSON, using defaults");
    return;
  }

  cJSON *item = nullptr;

  // Helper function to parse model parameters from either root or model object
  auto parse_model_params = [&](cJSON *config_obj) {
    if ((item = cJSON_GetObjectItem(config_obj, "n_predict")))
    {
      params.n_predict = (int32_t)cJSON_GetNumberValue(item);
    }

    if ((item = cJSON_GetObjectItem(config_obj, "n_gpu_layers")))
    {
      params.n_gpu_layers = (int32_t)cJSON_GetNumberValue(item);
    }

    if ((item = cJSON_GetObjectItem(config_obj, "ctx_size")))
    {
      params.n_ctx = (uint32_t)cJSON_GetNumberValue(item);
    }

    if ((item = cJSON_GetObjectItem(config_obj, "batch_size")))
    {
      params.n_batch = (uint32_t)cJSON_GetNumberValue(item);
    }

    if ((item = cJSON_GetObjectItem(config_obj, "threads")))
    {
      params.cpuparams.n_threads = (uint32_t)cJSON_GetNumberValue(item);
      params.cpuparams_batch.n_threads = (uint32_t)cJSON_GetNumberValue(item);
    }
  };

  // Parse model parameters - first check for new nested structure
  cJSON *model_config = cJSON_GetObjectItem(root, "model");
  if (cJSON_IsObject(model_config))
  {
    // New nested model configuration
    parse_model_params(model_config);
  }
  else
  {
    // Legacy flat configuration (backward compatibility)
    parse_model_params(root);
  }

  // Simple sampling parameters (backward compatibility)
  if ((item = cJSON_GetObjectItem(root, "temp")))
  {
    params.sampling.temp = (float)cJSON_GetNumberValue(item);
  }

  if ((item = cJSON_GetObjectItem(root, "top_p")))
  {
    params.sampling.top_p = (float)cJSON_GetNumberValue(item);
  }

  if ((item = cJSON_GetObjectItem(root, "repeat_penalty")))
  {
    params.sampling.penalty_repeat = (float)cJSON_GetNumberValue(item);
  }

  // Advanced sampling parameters
  cJSON *sampling = cJSON_GetObjectItem(root, "sampling");
  if (cJSON_IsObject(sampling))
  {
    cJSON *temp = cJSON_GetObjectItem(sampling, "temp");
    if (cJSON_IsNumber(temp))
    {
      params.sampling.temp = (float)cJSON_GetNumberValue(temp);
    }

    cJSON *top_p = cJSON_GetObjectItem(sampling, "top_p");
    if (cJSON_IsNumber(top_p))
    {
      params.sampling.top_p = (float)cJSON_GetNumberValue(top_p);
    }

    cJSON *top_k = cJSON_GetObjectItem(sampling, "top_k");
    if (cJSON_IsNumber(top_k))
    {
      params.sampling.top_k = (int32_t)cJSON_GetNumberValue(top_k);
    }

    cJSON *min_p = cJSON_GetObjectItem(sampling, "min_p");
    if (cJSON_IsNumber(min_p))
    {
      params.sampling.min_p = (float)cJSON_GetNumberValue(min_p);
    }

    // Note: tfs_z might not be available in this version
    // cJSON *tfs_z = cJSON_GetObjectItem(sampling, "tfs_z");
    // if (cJSON_IsNumber(tfs_z))
    // {
    //   params.sampling.tfs_z = (float)cJSON_GetNumberValue(tfs_z);
    // }

    cJSON *typical_p = cJSON_GetObjectItem(sampling, "typical_p");
    if (cJSON_IsNumber(typical_p))
    {
      params.sampling.typ_p = (float)cJSON_GetNumberValue(typical_p);
    }

    cJSON *repeat_penalty = cJSON_GetObjectItem(sampling, "repeat_penalty");
    if (cJSON_IsNumber(repeat_penalty))
    {
      params.sampling.penalty_repeat = (float)cJSON_GetNumberValue(repeat_penalty);
    }

    cJSON *presence_penalty = cJSON_GetObjectItem(sampling, "presence_penalty");
    if (cJSON_IsNumber(presence_penalty))
    {
      params.sampling.penalty_present = (float)cJSON_GetNumberValue(presence_penalty);
    }

    cJSON *frequency_penalty = cJSON_GetObjectItem(sampling, "frequency_penalty");
    if (cJSON_IsNumber(frequency_penalty))
    {
      params.sampling.penalty_freq = (float)cJSON_GetNumberValue(frequency_penalty);
    }

    cJSON *penalty_last_n = cJSON_GetObjectItem(sampling, "penalty_last_n");
    if (cJSON_IsNumber(penalty_last_n))
    {
      params.sampling.penalty_last_n = (int32_t)cJSON_GetNumberValue(penalty_last_n);
    }

    cJSON *mirostat = cJSON_GetObjectItem(sampling, "mirostat");
    if (cJSON_IsNumber(mirostat))
    {
      params.sampling.mirostat = (int32_t)cJSON_GetNumberValue(mirostat);
    }

    cJSON *mirostat_tau = cJSON_GetObjectItem(sampling, "mirostat_tau");
    if (cJSON_IsNumber(mirostat_tau))
    {
      params.sampling.mirostat_tau = (float)cJSON_GetNumberValue(mirostat_tau);
    }

    cJSON *mirostat_eta = cJSON_GetObjectItem(sampling, "mirostat_eta");
    if (cJSON_IsNumber(mirostat_eta))
    {
      params.sampling.mirostat_eta = (float)cJSON_GetNumberValue(mirostat_eta);
    }

    cJSON *seed = cJSON_GetObjectItem(sampling, "seed");
    if (cJSON_IsNumber(seed))
    {
      params.sampling.seed = (int32_t)cJSON_GetNumberValue(seed);
    }
  }

  // Stopping criteria
  cJSON *stopping = cJSON_GetObjectItem(root, "stopping");
  if (cJSON_IsObject(stopping))
  {
    cJSON *max_tokens = cJSON_GetObjectItem(stopping, "max_tokens");
    if (cJSON_IsNumber(max_tokens))
    {
      params.n_predict = (int32_t)cJSON_GetNumberValue(max_tokens);
    }

    // Use max_params instead of max_gen_time if available
    cJSON *max_time_ms = cJSON_GetObjectItem(stopping, "max_time_ms");
    if (cJSON_IsNumber(max_time_ms))
    {
      // Convert milliseconds to seconds
      // Note: This field might not exist in this version of the library
      // params.max_gen_time = (float)cJSON_GetNumberValue(max_time_ms) / 1000.0f;
    }

    cJSON *ignore_eos = cJSON_GetObjectItem(stopping, "ignore_eos");
    if (cJSON_IsBool(ignore_eos))
    {
      params.sampling.ignore_eos = cJSON_IsTrue(ignore_eos);
    }

    // Handle stop sequences
    cJSON *stop = cJSON_GetObjectItem(stopping, "stop");
    if (cJSON_IsArray(stop))
    {
      // Clear existing stop sequences
      params.sampling.grammar_triggers.clear();
      
      int array_size = cJSON_GetArraySize(stop);
      for (int i = 0; i < array_size; i++)
      {
        cJSON *stop_item = cJSON_GetArrayItem(stop, i);
        if (cJSON_IsString(stop_item))
        {
          // Add stop sequence to grammar triggers
          common_grammar_trigger trigger;
          trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN;  // Use pattern type for string matching
          trigger.value = std::string(cJSON_GetStringValue(stop_item));
          params.sampling.grammar_triggers.push_back(trigger);
        }
      }
    }
  }

  // Memory management
  cJSON *memory = cJSON_GetObjectItem(root, "memory");
  if (cJSON_IsObject(memory))
  {
    cJSON *context_shifting = cJSON_GetObjectItem(memory, "context_shifting");
    if (cJSON_IsBool(context_shifting))
    {
      // This will be handled at the context level
    }

    // Use path_prompt_cache instead of cache_prompt if available
    cJSON *cache_prompt = cJSON_GetObjectItem(memory, "cache_prompt");
    if (cJSON_IsBool(cache_prompt))
    {
      // Note: This field might not exist in this version of the library
      // params.cache_prompt = cJSON_IsTrue(cache_prompt);
    }
  }

  cJSON_Delete(root);
}

// Helper function to setup threadpools (from main.cpp)
static wasi_nn_error setup_threadpools(LlamaChatContext *chat_ctx)
{
  // Access server context members through server_ctx
  common_params& params = chat_ctx->server_ctx.params_base;
  
  auto *reg = ggml_backend_dev_backend_reg(
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
  auto ggml_threadpool_new_fn =
      (decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_new");
  auto ggml_threadpool_free_fn =
      (decltype(ggml_threadpool_free) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_free");

  struct ggml_threadpool_params tpp_batch =
      ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
  struct ggml_threadpool_params tpp =
      ggml_threadpool_params_from_cpu_params(params.cpuparams);

  // Create batch threadpool if different from main threadpool
  ggml_threadpool* threadpool_batch = nullptr;
  if (!ggml_threadpool_params_match(&tpp, &tpp_batch))
  {
    threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
    if (!threadpool_batch)
    {
      NN_ERR_PRINTF("Failed to create batch threadpool");
      return runtime_error;
    }
    tpp.paused = true;
  }

  ggml_threadpool* threadpool = ggml_threadpool_new_fn(&tpp);
  if (!threadpool)
  {
    NN_ERR_PRINTF("Failed to create threadpool");
    return runtime_error;
  }

  llama_attach_threadpool(chat_ctx->server_ctx.ctx, threadpool,
                          threadpool_batch);
  return success;
}

// Function to safely copy a string into tensor_data (from original)
void copy_string_to_tensor_data(tensor_data dest, uint32_t dest_size,
                                const std::string &src)
{
  if (dest == nullptr || dest_size == 0)
  {
    NN_ERR_PRINTF("Destination buffer is null or size is zero");
    return;
  }

  size_t src_size = src.size();
  if (src_size >= dest_size)
  {
    NN_WARN_PRINTF("Source string is too long, truncating");
    src_size = dest_size - 1;
  }

  std::copy(src.begin(), src.begin() + src_size,
            reinterpret_cast<uint8_t *>(dest));
  if (dest_size > 0)
  {
    dest[src_size] = '\0';
  }
}

// Main API functions
__attribute__((visibility("default"))) wasi_nn_error init_backend(void **ctx)
{
  return init_backend_with_config(ctx, nullptr, 0);
}

__attribute__((visibility("default"))) wasi_nn_error
init_backend_with_config(void **ctx, const char *config, uint32_t config_len)
{
  LlamaChatContext *chat_ctx = new LlamaChatContext();
  if (!chat_ctx)
  {
    NN_ERR_PRINTF("Failed to allocate chat context");
    return runtime_error;
  }

  // Parse config JSON to update settings if provided
  if (config && config_len > 0)
  {
    cJSON *json = cJSON_ParseWithLength(config, config_len);
    if (json)
    {
      // Helper function to parse backend configuration from either root or backend object
      auto parse_backend_config = [&](cJSON *config_obj) {
        cJSON *max_sessions = cJSON_GetObjectItem(config_obj, "max_sessions");
        if (cJSON_IsNumber(max_sessions))
        {
          chat_ctx->max_sessions = (uint32_t)max_sessions->valueint;
        }

        cJSON *idle_timeout = cJSON_GetObjectItem(config_obj, "idle_timeout_ms");
        if (cJSON_IsNumber(idle_timeout))
        {
          chat_ctx->idle_timeout_ms = (uint32_t)idle_timeout->valueint;
        }

        cJSON *auto_cleanup = cJSON_GetObjectItem(config_obj, "auto_cleanup");
        if (cJSON_IsBool(auto_cleanup))
        {
          chat_ctx->auto_cleanup_enabled = cJSON_IsTrue(auto_cleanup);
        }

        cJSON *max_concurrent = cJSON_GetObjectItem(config_obj, "max_concurrent");
        if (cJSON_IsNumber(max_concurrent))
        {
          chat_ctx->max_concurrent = (uint32_t)max_concurrent->valueint;
        }

        cJSON *queue_size = cJSON_GetObjectItem(config_obj, "queue_size");
        if (cJSON_IsNumber(queue_size))
        {
          chat_ctx->queue_size = (uint32_t)queue_size->valueint;
        }
      };

      // Parse backend configuration - first check for new nested structure
      cJSON *backend_config = cJSON_GetObjectItem(json, "backend");
      if (cJSON_IsObject(backend_config))
      {
        // New nested backend configuration
        parse_backend_config(backend_config);
      }
      else
      {
        // Legacy flat configuration (backward compatibility)
        parse_backend_config(json);
      }

      // Memory policy
      cJSON *memory_policy = cJSON_GetObjectItem(json, "memory_policy");
      if (cJSON_IsObject(memory_policy))
      {
        cJSON *context_shifting = cJSON_GetObjectItem(memory_policy, "context_shifting");
        if (cJSON_IsBool(context_shifting))
        {
          chat_ctx->context_shifting_enabled = cJSON_IsTrue(context_shifting);
        }

        cJSON *cache_strategy = cJSON_GetObjectItem(memory_policy, "cache_strategy");
        if (cJSON_IsString(cache_strategy))
        {
          chat_ctx->cache_strategy = std::string(cJSON_GetStringValue(cache_strategy));
        }

        cJSON *max_cache_tokens = cJSON_GetObjectItem(memory_policy, "max_cache_tokens");
        if (cJSON_IsNumber(max_cache_tokens))
        {
          chat_ctx->max_cache_tokens = (uint32_t)max_cache_tokens->valueint;
        }
      }

      // Logging configuration
      cJSON *logging = cJSON_GetObjectItem(json, "logging");
      if (cJSON_IsObject(logging))
      {
        cJSON *log_level = cJSON_GetObjectItem(logging, "level");
        if (cJSON_IsString(log_level))
        {
          chat_ctx->log_level = std::string(cJSON_GetStringValue(log_level));
        }

        cJSON *enable_debug = cJSON_GetObjectItem(logging, "enable_debug");
        if (cJSON_IsBool(enable_debug))
        {
          chat_ctx->enable_debug_log = cJSON_IsTrue(enable_debug);
        }

        cJSON *log_file = cJSON_GetObjectItem(logging, "file");
        if (cJSON_IsString(log_file))
        {
          chat_ctx->log_file = std::string(cJSON_GetStringValue(log_file));
        }
      }

      // Performance settings
      cJSON *performance = cJSON_GetObjectItem(json, "performance");
      if (cJSON_IsObject(performance))
      {
        cJSON *batch_processing = cJSON_GetObjectItem(performance, "batch_processing");
        if (cJSON_IsBool(batch_processing))
        {
          chat_ctx->batch_processing_enabled = cJSON_IsTrue(batch_processing);
        }

        cJSON *batch_size = cJSON_GetObjectItem(performance, "batch_size");
        if (cJSON_IsNumber(batch_size))
        {
          chat_ctx->batch_size = (uint32_t)batch_size->valueint;
        }
      }

      cJSON_Delete(json);
    }
  }

  // Initialize llama backend (from main.cpp)
  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  NN_INFO_PRINTF("Llama chat backend initialized successfully");
  NN_INFO_PRINTF(
      "Session config: max_sessions=%d, idle_timeout_ms=%d, auto_cleanup=%s",
      chat_ctx->max_sessions, chat_ctx->idle_timeout_ms,
      chat_ctx->auto_cleanup_enabled ? "true" : "false");
  NN_INFO_PRINTF(
      "Concurrency config: max_concurrent=%d, queue_size=%d",
      chat_ctx->max_concurrent, chat_ctx->queue_size);
  NN_INFO_PRINTF(
      "Memory config: context_shifting=%s, cache_strategy=%s, max_cache_tokens=%d",
      chat_ctx->context_shifting_enabled ? "true" : "false",
      chat_ctx->cache_strategy.c_str(), chat_ctx->max_cache_tokens);
  NN_INFO_PRINTF(
      "Logging config: level=%s, enable_debug=%s, file=%s",
      chat_ctx->log_level.c_str(),
      chat_ctx->enable_debug_log ? "true" : "false",
      chat_ctx->log_file.c_str());
  NN_INFO_PRINTF(
      "Performance config: batch_processing=%s, batch_size=%d",
      chat_ctx->batch_processing_enabled ? "true" : "false",
      chat_ctx->batch_size);
  *ctx = (void *)chat_ctx;
  return success;
}

__attribute__((visibility("default"))) wasi_nn_error deinit_backend(void *ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  // Note: model and ctx are managed by common_init_result's unique_ptrs
  // They will be automatically cleaned up by the server_context

  llama_backend_free();
  delete chat_ctx;
  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
load_by_name_with_config(void *ctx, const char *filename, uint32_t filename_len,
                         const char *config, uint32_t config_len, graph *g)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  NN_DBG_PRINTF("Loading model: %s", filename);
  NN_DBG_PRINTF("Config: %s", config ? config : "null");

  // Parse config into params
  parse_config_to_params(config, chat_ctx->server_ctx.params_base);
  chat_ctx->server_ctx.params_base.model.path = filename;

  NN_INFO_PRINTF("Model config: n_gpu_layers=%d, ctx_size=%d, batch_size=%d, threads=%d",
                 chat_ctx->server_ctx.params_base.n_gpu_layers,
                 chat_ctx->server_ctx.params_base.n_ctx,
                 chat_ctx->server_ctx.params_base.n_batch,
                 chat_ctx->server_ctx.params_base.cpuparams.n_threads);

  // Load model using server_context's approach
  if (!chat_ctx->server_ctx.load_model(chat_ctx->server_ctx.params_base)) {
      NN_ERR_PRINTF("Failed to load model from file %s", filename);
      return runtime_error;
  }

  // Check context size
  const int n_ctx_train = llama_model_n_ctx_train(chat_ctx->server_ctx.model);
  const int n_ctx = llama_n_ctx(chat_ctx->server_ctx.ctx);

  if (n_ctx > n_ctx_train)
  {
    NN_WARN_PRINTF("Model was trained on only %d context tokens (%d specified)",
                   n_ctx_train, n_ctx);
  }

  NN_INFO_PRINTF("Model loaded successfully. Context size: %d", n_ctx);

  return success;
}

// Auto-cleanup function: removes old/excess sessions
static void auto_cleanup_sessions(LlamaChatContext *chat_ctx)
{
  if (!chat_ctx->auto_cleanup_enabled)
    return;

  auto now = std::chrono::steady_clock::now();
  auto idle_timeout = std::chrono::milliseconds(chat_ctx->idle_timeout_ms);

  // Remove idle sessions
  for (auto it = chat_ctx->sessions.begin(); it != chat_ctx->sessions.end();)
  {
    if ((now - it->second.last_activity) > idle_timeout)
    {
      auto idle_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - it->second.last_activity)
                         .count();
      NN_INFO_PRINTF("Auto-cleanup: removing idle session %d (idle for %lldms)",
                     it->first, (long long)idle_time);
      it = chat_ctx->sessions.erase(it);
    }
    else
    {
      ++it;
    }
  }

  // Remove excess sessions (LRU eviction)
  if (chat_ctx->sessions.size() >= chat_ctx->max_sessions)
  {
    // Sort by last_activity and remove the oldest
    std::vector<std::pair<graph_execution_context,
                          std::chrono::steady_clock::time_point>>
        sorted_sessions;
    for (const auto &session : chat_ctx->sessions)
    {
      sorted_sessions.emplace_back(session.first, session.second.last_activity);
    }

    std::sort(sorted_sessions.begin(), sorted_sessions.end(),
              [](const auto &a, const auto &b)
              { return a.second < b.second; });

    // Remove oldest sessions to make room
    size_t sessions_to_remove =
        chat_ctx->sessions.size() - chat_ctx->max_sessions + 1;
    for (size_t i = 0; i < sessions_to_remove && i < sorted_sessions.size();
         ++i)
    {
      auto exec_ctx_id = sorted_sessions[i].first;
      NN_INFO_PRINTF("Auto-cleanup: removing session %d (max sessions reached)",
                     exec_ctx_id);
      chat_ctx->sessions.erase(exec_ctx_id);
    }
  }
}

__attribute__((visibility("default"))) wasi_nn_error init_execution_context(
    void *ctx, graph g, graph_execution_context *exec_ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx || !chat_ctx->server_ctx.model)
    return invalid_argument;

  // Check concurrency limit
  if (chat_ctx->active_sessions + 1 > chat_ctx->max_concurrent)
  {
    NN_ERR_PRINTF("Concurrency limit reached: %d active sessions, max allowed: %d",
                  chat_ctx->active_sessions, chat_ctx->max_concurrent);
    return runtime_error;
  }

  // Auto-cleanup on entry
  auto_cleanup_sessions(chat_ctx);

  // Initialize sampler if not already done (from main.cpp)
  // Setup threadpools if not already done
  wasi_nn_error result = setup_threadpools(chat_ctx);
  if (result != success)
  {
    return result;
  }

  // Initialize the server context
  chat_ctx->server_ctx.init();

  // Initialize samplers for all slots (crucial for inference)
  for (auto &slot : chat_ctx->server_ctx.slots) {
    if (slot.smpl != nullptr) {
      common_sampler_free(slot.smpl);
    }
    slot.smpl = common_sampler_init(chat_ctx->server_ctx.model, slot.params.sampling);
    if (slot.smpl == nullptr) {
      NN_ERR_PRINTF("Failed to initialize sampler for slot %d", slot.id);
      return runtime_error;
    }
  }

  // Create new session
  graph_execution_context new_exec_ctx = chat_ctx->next_exec_ctx_id++;
  SessionInfo session_info;
  session_info.session_id = "session_" + std::to_string(new_exec_ctx);
  session_info.last_activity = std::chrono::steady_clock::now();

  chat_ctx->sessions[new_exec_ctx] = std::move(session_info);
  chat_ctx->active_sessions++; // Increment active sessions counter

  *exec_ctx = new_exec_ctx;

  NN_INFO_PRINTF(
      "Execution context %d initialized. Active sessions: %d, Max concurrent: %d",
      new_exec_ctx, chat_ctx->active_sessions, chat_ctx->max_concurrent);

  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
close_execution_context(void *ctx, graph_execution_context exec_ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  auto it = chat_ctx->sessions.find(exec_ctx);
  if (it != chat_ctx->sessions.end())
  {
    NN_INFO_PRINTF("Closing execution context %d for session '%s'", exec_ctx,
                   it->second.session_id.c_str());
    chat_ctx->sessions.erase(it);
    if (chat_ctx->active_sessions > 0)
    {
      chat_ctx->active_sessions--; // Decrement active sessions counter
    }
    return success;
  }

  return invalid_argument;
}

// Helper function to run inference loop (extracted from main.cpp)
static std::string run_inference_for_session(LlamaChatContext *chat_ctx,
                                             graph_execution_context exec_ctx,
                                             const std::string &user_input)
{
  // Find session
  auto session_it = chat_ctx->sessions.find(exec_ctx);
  if (session_it == chat_ctx->sessions.end())
  {
    return "Error: Invalid session";
  }

  SessionInfo &session_info = session_it->second;
  auto &chat_msgs = session_info.chat_history;

  // Update last activity
  session_info.last_activity = std::chrono::steady_clock::now();

  // Chat formatting function (from main.cpp)
  auto chat_add_and_format = [&](const std::string &role,
                                 const std::string &content)
  {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;

    // Check if chat templates are available
    if (!chat_ctx->server_ctx.chat_templates.get()) {
      NN_ERR_PRINTF("Chat templates not initialized");
      return std::string("Error: Chat templates not available");
    }

    auto formatted = common_chat_format_single(
        chat_ctx->server_ctx.chat_templates.get(), chat_msgs, new_msg, role == "user",
        false // use_jinja
    );

    chat_msgs.push_back(new_msg);
    NN_DBG_PRINTF("Formatted message: '%s'", formatted.c_str());
    return formatted;
  };

  // Add user message and get formatted prompt
  std::string prompt = chat_add_and_format("user", user_input);

  NN_DBG_PRINTF("Processing prompt for session %d: %s", exec_ctx,
                prompt.c_str());

  // Clear KV cache for session isolation (as per user's requirement)
  llama_memory_clear(llama_get_memory(chat_ctx->server_ctx.ctx), true);

  // Tokenize the complete conversation history
  common_chat_templates_inputs inputs;
  inputs.messages = chat_msgs;
  inputs.add_generation_prompt = true;

  // Check if chat templates are available before using them
  if (!chat_ctx->server_ctx.chat_templates.get()) {
    NN_ERR_PRINTF("Chat templates not initialized for prompt generation");
    return "Error: Chat templates not available";
  }

  std::string full_prompt =
      common_chat_templates_apply(chat_ctx->server_ctx.chat_templates.get(), inputs)
          .prompt;

  // Tokenize
  std::vector<llama_token> tokens =
      common_tokenize(chat_ctx->server_ctx.ctx, full_prompt, true, true);

  // Generate response (simplified version of main.cpp's loop)
  std::string response;

  // Process input tokens
  llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

  if (llama_decode(chat_ctx->server_ctx.ctx, batch))
  {
    NN_ERR_PRINTF("Failed to decode input tokens");
    return "Error: Failed to process input";
  }

  // Generate tokens one by one
  for (int i = 0; i < chat_ctx->server_ctx.params_base.n_predict; ++i)
  {
    // Verify that slots[0] and its sampler are valid
    if (chat_ctx->server_ctx.slots.empty() || chat_ctx->server_ctx.slots[0].smpl == nullptr) {
      NN_ERR_PRINTF("Invalid slot or sampler state");
      return "Error: Invalid sampler state";
    }

    llama_token new_token =
        common_sampler_sample(chat_ctx->server_ctx.slots[0].smpl, chat_ctx->server_ctx.ctx, -1);

    if (llama_vocab_is_eog(chat_ctx->server_ctx.vocab, new_token))
    {
      break;
    }

    // Convert token to text
    char buf[256];
    int n = llama_token_to_piece(chat_ctx->server_ctx.vocab, new_token, buf, sizeof(buf),
                                 0, true);
    if (n > 0)
    {
      response.append(buf, n);
    }

    // Prepare next batch
    batch = llama_batch_get_one(&new_token, 1);
    if (llama_decode(chat_ctx->server_ctx.ctx, batch))
    {
      NN_ERR_PRINTF("Failed to decode generated token");
      break;
    }
  }

  // Add assistant response to chat history
  chat_add_and_format("assistant", response);

  return response;
}

__attribute__((visibility("default"))) wasi_nn_error
run_inference(void *ctx, graph_execution_context exec_ctx, uint32_t index,
              tensor *input_tensor, tensor_data output_tensor,
              uint32_t *output_tensor_size)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx || !chat_ctx->server_ctx.ctx)
  {
    return invalid_argument;
  }

  char *prompt_text = (char *)input_tensor->data;
  if (!prompt_text)
  {
    return invalid_argument;
  }

  try
  {
    std::string response =
        run_inference_for_session(chat_ctx, exec_ctx, prompt_text);

    *output_tensor_size = response.size() + 1;
    copy_string_to_tensor_data(output_tensor, *output_tensor_size, response);

    NN_DBG_PRINTF("Generated response: %s", response.c_str());
    return success;
  }
  catch (const std::exception &e)
  {
    NN_ERR_PRINTF("Inference failed: %s", e.what());
    return runtime_error;
  }
}

// Placeholder implementations for compatibility
__attribute__((visibility("default"))) wasi_nn_error
load(void *ctx, graph_builder_array *builder, graph_encoding encoding,
     execution_target target, graph *g)
{
  return unsupported_operation;
}

__attribute__((visibility("default"))) wasi_nn_error
load_by_name(void *ctx, const char *filename, uint32_t filename_len, graph *g)
{
  // Use default config
  return load_by_name_with_config(ctx, filename, filename_len, nullptr, 0, g);
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
