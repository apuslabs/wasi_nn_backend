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
#include <deque>
#include <thread>
#include <condition_variable>

// Task priority levels for WASI-NN backend
enum wasi_nn_task_priority
{
  WASI_NN_PRIORITY_LOW = 0,
  WASI_NN_PRIORITY_NORMAL = 1,
  WASI_NN_PRIORITY_HIGH = 2,
  WASI_NN_PRIORITY_URGENT = 3
};

// Enhanced task structure for WASI-NN backend
struct wasi_nn_task
{
  int id = -1;
  graph_execution_context exec_ctx;
  wasi_nn_task_priority priority = WASI_NN_PRIORITY_NORMAL;
  std::chrono::steady_clock::time_point created_at;
  std::chrono::steady_clock::time_point timeout_at;
  uint32_t timeout_ms = 30000; // Default 30 second timeout
  std::string prompt;
  bool is_queued = false;
  
  wasi_nn_task() : created_at(std::chrono::steady_clock::now()) 
  {
    timeout_at = created_at + std::chrono::milliseconds(timeout_ms);
  }
};

// Forward declaration for task queue
struct wasi_nn_task_queue;

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

  // Enhanced concurrency and task management (Phase 4.2)
  uint32_t max_concurrent;
  uint32_t queue_size;
  uint32_t active_sessions; // Track active sessions
  
  // Advanced task queue system
  std::shared_ptr<wasi_nn_task_queue> task_queue;
  std::thread task_processor_thread;
  bool task_processing_enabled = true;
  
  // Task timeout and priority settings
  uint32_t default_task_timeout_ms = 30000;
  bool priority_scheduling_enabled = true;
  bool fair_scheduling_enabled = true;
  
  // Queue monitoring and limits
  uint32_t queue_warning_threshold = 40;    // Warn when queue is 80% full
  uint32_t queue_reject_threshold = 50;     // Reject when queue is 100% full
  bool auto_queue_cleanup = true;

  // Memory policy
  bool context_shifting_enabled;
  std::string cache_strategy;
  uint32_t max_cache_tokens;
  
  // Phase 4.3: Advanced Memory Management
  uint32_t n_keep_tokens = 256;             // Number of tokens to keep when shifting context
  uint32_t n_discard_tokens = 0;            // Number of tokens to discard (0 = auto half)
  float memory_pressure_threshold = 0.85f;  // Trigger cleanup at 85% memory usage
  bool enable_partial_cache_deletion = true;
  bool enable_token_cache_reuse = true;
  std::string cache_deletion_strategy = "lru";  // lru, fifo, or smart
  uint32_t max_memory_mb = 0;               // 0 = no limit
  
  // Memory monitoring
  std::atomic<uint64_t> current_memory_usage{0};
  std::atomic<uint64_t> peak_memory_usage{0};
  std::atomic<uint32_t> cache_hits{0};
  std::atomic<uint32_t> cache_misses{0};

  // Logging configuration
  std::string log_level;
  bool enable_debug_log;
  std::string log_file;
  bool enable_timestamps;
  bool enable_colors;
  
  // Logging system state
  struct common_log * log_instance;
  bool log_initialized;

  // Performance settings
  bool batch_processing_enabled;
  uint32_t batch_size;

  LlamaChatContext()
      : next_exec_ctx_id(1),
        max_sessions(100), idle_timeout_ms(300000), auto_cleanup_enabled(true),
        max_concurrent(8), queue_size(50), active_sessions(0),
        context_shifting_enabled(true), cache_strategy("lru"), max_cache_tokens(10000),
        n_keep_tokens(256), n_discard_tokens(0), memory_pressure_threshold(0.85f),
        enable_partial_cache_deletion(true), enable_token_cache_reuse(true),
        cache_deletion_strategy("lru"), max_memory_mb(0),
        current_memory_usage(0), peak_memory_usage(0), cache_hits(0), cache_misses(0),
        log_level("info"), enable_debug_log(false), enable_timestamps(true), enable_colors(false),
        log_instance(nullptr), log_initialized(false),
        batch_processing_enabled(true), batch_size(512) {}
  
  // Destructor will be defined after wasi_nn_task_queue definition
  ~LlamaChatContext();
};

// Task queue with priority management
struct wasi_nn_task_queue
{
  std::deque<wasi_nn_task> high_priority_queue;    // Priority 3 (urgent)
  std::deque<wasi_nn_task> normal_priority_queue;  // Priority 1-2 (normal/high)
  std::deque<wasi_nn_task> low_priority_queue;     // Priority 0 (low)
  
  std::mutex queue_mutex;
  std::condition_variable queue_condition;
  
  uint32_t max_queue_size = 50;
  uint32_t current_size = 0;
  bool running = true;
  int next_task_id = 1;
  
  // Queue statistics
  uint32_t tasks_queued = 0;
  uint32_t tasks_completed = 0;
  uint32_t tasks_timeout = 0;
  uint32_t tasks_rejected = 0;
  
  // Add task to appropriate priority queue
  bool enqueue_task(wasi_nn_task &&task, LlamaChatContext* ctx = nullptr);
  
  // Get next task based on priority
  bool dequeue_task(wasi_nn_task &task, LlamaChatContext* ctx = nullptr);
  
  // Clean up expired tasks
  void cleanup_expired_tasks();
  
  // Get queue status
  void get_queue_status(uint32_t &queued, uint32_t &active, uint32_t &capacity);
};

// Implementation of LlamaChatContext destructor
LlamaChatContext::~LlamaChatContext() {
  // Cleanup logging system
  if (log_initialized && log_instance) {
    common_log_free(log_instance);
    log_instance = nullptr;
    log_initialized = false;
  }
  
  // Cleanup task processing thread
  if (task_processing_enabled && task_processor_thread.joinable()) {
    if (task_queue) {
      task_queue->running = false;
      task_queue->queue_condition.notify_all();
    }
    task_processor_thread.join();
  }
}

// ==============================================================================
// Phase 5.1: Advanced Logging System Implementation
// ==============================================================================

// Convert string log level to common_log verbosity level
static int string_to_log_verbosity(const std::string& level) {
  if (level == "debug" || level == "DEBUG") return 0;
  if (level == "info" || level == "INFO") return 1;
  if (level == "warn" || level == "warning" || level == "WARN" || level == "WARNING") return 2;
  if (level == "error" || level == "ERROR") return 2;
  if (level == "none" || level == "NONE" || level == "off" || level == "OFF") return 4;
  return 1; // Default to INFO level
}

// Initialize advanced logging system
static bool initialize_advanced_logging(LlamaChatContext* chat_ctx) {
  if (!chat_ctx) return false;
  
  // Initialize llama.cpp logging system
  if (chat_ctx->log_instance) {
    common_log_free(chat_ctx->log_instance);
  }
  
  chat_ctx->log_instance = common_log_init();
  if (!chat_ctx->log_instance) {
    NN_ERR_PRINTF("Failed to initialize advanced logging system");
    return false;
  }
  
  // Set logging verbosity based on configuration
  int verbosity = string_to_log_verbosity(chat_ctx->log_level);
  common_log_set_verbosity_thold(verbosity);
  
  // Configure colors
  common_log_set_colors(chat_ctx->log_instance, chat_ctx->enable_colors);
  
  // Configure timestamps and prefixes
  common_log_set_timestamps(chat_ctx->log_instance, chat_ctx->enable_timestamps);
  common_log_set_prefix(chat_ctx->log_instance, true);
  
  // Configure file output if specified
  if (!chat_ctx->log_file.empty()) {
    common_log_set_file(chat_ctx->log_instance, chat_ctx->log_file.c_str());
  }
  
  chat_ctx->log_initialized = true;
  
  // Log system initialization success
  LOG_INF("Advanced logging system initialized");
  LOG_INF("Log level: %s (verbosity: %d)", chat_ctx->log_level.c_str(), verbosity);
  LOG_INF("Debug mode: %s", chat_ctx->enable_debug_log ? "enabled" : "disabled");
  LOG_INF("Colors: %s", chat_ctx->enable_colors ? "enabled" : "disabled");
  LOG_INF("Timestamps: %s", chat_ctx->enable_timestamps ? "enabled" : "disabled");
  if (!chat_ctx->log_file.empty()) {
    LOG_INF("File logging: %s", chat_ctx->log_file.c_str());
  }
  
  return true;
}

// Enhanced logging macros that work with both old and new systems
#define WASI_NN_LOG_DEBUG(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_DBG("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_DBG_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

#define WASI_NN_LOG_INFO(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_INF("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_INFO_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

#define WASI_NN_LOG_WARN(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_WRN("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_WARN_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

#define WASI_NN_LOG_ERROR(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_ERR("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_ERR_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

// Structured logging for performance metrics
static void log_performance_metrics(LlamaChatContext* chat_ctx, const std::string& operation, 
                                   int64_t duration_us, const std::string& additional_info = "") {
  if (!chat_ctx || !chat_ctx->log_initialized) return;
  
  double duration_ms = duration_us / 1000.0;
  if (additional_info.empty()) {
    LOG_INF("[PERF] %s completed in %.2fms", operation.c_str(), duration_ms);
  } else {
    LOG_INF("[PERF] %s completed in %.2fms - %s", operation.c_str(), duration_ms, additional_info.c_str());
  }
}

// Structured logging for memory operations
static void log_memory_operation(LlamaChatContext* chat_ctx, const std::string& operation, 
                                uint64_t memory_used, uint64_t memory_limit = 0) {
  if (!chat_ctx || !chat_ctx->log_initialized) return;
  
  double memory_mb = memory_used / (1024.0 * 1024.0);
  if (memory_limit > 0) {
    double limit_mb = memory_limit / (1024.0 * 1024.0);
    double usage_percent = (memory_used * 100.0) / memory_limit;
    LOG_INF("[MEM] %s - %.2fMB used (%.1f%% of %.2fMB limit)", 
            operation.c_str(), memory_mb, usage_percent, limit_mb);
  } else {
    LOG_INF("[MEM] %s - %.2fMB used", operation.c_str(), memory_mb);
  }
}

// Structured logging for task queue operations
static void log_task_operation(LlamaChatContext* chat_ctx, const std::string& operation,
                              int task_id, wasi_nn_task_priority priority = WASI_NN_PRIORITY_NORMAL,
                              const std::string& additional_info = "") {
  if (!chat_ctx || !chat_ctx->log_initialized) return;
  
  const char* priority_str = "NORMAL";
  switch (priority) {
    case WASI_NN_PRIORITY_LOW: priority_str = "LOW"; break;
    case WASI_NN_PRIORITY_NORMAL: priority_str = "NORMAL"; break;
    case WASI_NN_PRIORITY_HIGH: priority_str = "HIGH"; break;
    case WASI_NN_PRIORITY_URGENT: priority_str = "URGENT"; break;
  }
  
  if (additional_info.empty()) {
    LOG_INF("[TASK] %s - Task %d (Priority: %s)", operation.c_str(), task_id, priority_str);
  } else {
    LOG_INF("[TASK] %s - Task %d (Priority: %s) - %s", 
            operation.c_str(), task_id, priority_str, additional_info.c_str());
  }
}

// Implementation of wasi_nn_task_queue methods
bool wasi_nn_task_queue::enqueue_task(wasi_nn_task &&task, LlamaChatContext* ctx)
{
  std::unique_lock<std::mutex> lock(queue_mutex);
  
  // Check if queue is at capacity
  if (current_size >= max_queue_size) {
    tasks_rejected++;
    if (ctx) {
      WASI_NN_LOG_WARN(ctx, "Task queue full (%d/%d), rejecting task %d", 
                       current_size, max_queue_size, task.id);
    } else {
      NN_WARN_PRINTF("Task queue full (%d/%d), rejecting task %d", 
                     current_size, max_queue_size, task.id);
    }
    return false;
  }
  
  // Assign task ID if not set
  if (task.id == -1) {
    task.id = next_task_id++;
  }
  
  // Add to appropriate priority queue
  switch (task.priority) {
    case WASI_NN_PRIORITY_URGENT:
      high_priority_queue.push_back(std::move(task));
      break;
    case WASI_NN_PRIORITY_HIGH:
    case WASI_NN_PRIORITY_NORMAL:
      normal_priority_queue.push_back(std::move(task));
      break;
    case WASI_NN_PRIORITY_LOW:
    default:
      low_priority_queue.push_back(std::move(task));
      break;
  }
  
  current_size++;
  tasks_queued++;
  
  // Use advanced logging if available
  if (ctx) {
    log_task_operation(ctx, "Task Queued", task.id, task.priority, 
                      "Queue: " + std::to_string(current_size) + "/" + std::to_string(max_queue_size));
  } else {
    NN_INFO_PRINTF("Task %d queued with priority %d. Queue size: %d/%d", 
                   task.id, (int)task.priority, current_size, max_queue_size);
  }
  
  // Notify waiting threads
  queue_condition.notify_one();
  return true;
}

bool wasi_nn_task_queue::dequeue_task(wasi_nn_task &task, LlamaChatContext* ctx)
{
  std::unique_lock<std::mutex> lock(queue_mutex);
  
  // Wait for tasks to become available
  queue_condition.wait(lock, [this] { 
    return !running || 
           !high_priority_queue.empty() || 
           !normal_priority_queue.empty() || 
           !low_priority_queue.empty(); 
  });
  
  if (!running) {
    return false;
  }
  
  // Clean up expired tasks first
  cleanup_expired_tasks();
  
  // Dequeue from highest priority queue first
  if (!high_priority_queue.empty()) {
    task = std::move(high_priority_queue.front());
    high_priority_queue.pop_front();
  } else if (!normal_priority_queue.empty()) {
    task = std::move(normal_priority_queue.front());
    normal_priority_queue.pop_front();
  } else if (!low_priority_queue.empty()) {
    task = std::move(low_priority_queue.front());
    low_priority_queue.pop_front();
  } else {
    return false; // No tasks available
  }
  
  current_size--;
  
  // Use advanced logging if available
  if (ctx) {
    log_task_operation(ctx, "Task Dequeued", task.id, task.priority,
                      "Queue: " + std::to_string(current_size) + "/" + std::to_string(max_queue_size));
  } else {
    NN_INFO_PRINTF("Dequeued task %d with priority %d. Queue size: %d/%d",
                   task.id, (int)task.priority, current_size, max_queue_size);
  }
  
  return true;
}

void wasi_nn_task_queue::cleanup_expired_tasks()
{
  // Note: This method assumes the queue_mutex is already locked
  auto now = std::chrono::steady_clock::now();
  
  auto cleanup_queue = [&](std::deque<wasi_nn_task> &queue) {
    auto it = queue.begin();
    while (it != queue.end()) {
      if (now > it->timeout_at) {
        NN_WARN_PRINTF("Task %d expired (created %ldms ago)", 
                       it->id,
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - it->created_at).count());
        it = queue.erase(it);
        current_size--;
        tasks_timeout++;
      } else {
        ++it;
      }
    }
  };
  
  cleanup_queue(high_priority_queue);
  cleanup_queue(normal_priority_queue);
  cleanup_queue(low_priority_queue);
}

void wasi_nn_task_queue::get_queue_status(uint32_t &queued, uint32_t &active, uint32_t &capacity)
{
  std::unique_lock<std::mutex> lock(queue_mutex);
  queued = current_size;
  active = tasks_queued - tasks_completed - tasks_timeout - tasks_rejected;
  capacity = max_queue_size;
}

// Phase 4.3: Advanced Memory Management Functions
// ================================================

// Memory monitoring and pressure detection
static uint64_t get_current_memory_usage() {
  // This is a simplified implementation - in practice, you'd use system-specific calls
  // For now, we'll estimate based on context size and active sessions
  return 0; // Placeholder
}

static bool check_memory_pressure(LlamaChatContext* chat_ctx) {
  if (chat_ctx->max_memory_mb == 0) {
    return false; // No memory limit set
  }
  
  uint64_t current_mb = chat_ctx->current_memory_usage.load() / (1024 * 1024);
  uint64_t max_mb = chat_ctx->max_memory_mb;
  float usage_ratio = (float)current_mb / max_mb;
  
  return usage_ratio >= chat_ctx->memory_pressure_threshold;
}

// Context shifting implementation based on server.cpp
static wasi_nn_error perform_context_shift(LlamaChatContext* chat_ctx, uint32_t session_id) {
  if (!chat_ctx->context_shifting_enabled) {
    NN_ERR_PRINTF("Context shifting is disabled");
    return runtime_error;
  }
  
  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;
  
  if (!ctx) {
    NN_ERR_PRINTF("No context available for shifting");
    return runtime_error;
  }
  
  const int n_ctx = llama_n_ctx(ctx);
  const int n_keep = chat_ctx->n_keep_tokens;
  
  // In server.cpp, n_past is tracked per slot - here we use a simplified approach
  // For a full implementation, you would track n_past per session
  int n_past = n_ctx * 0.8f; // Assume 80% filled as a simplified estimate
  const int n_left = n_past - n_keep;
  
  if (n_left <= 0) {
    NN_WARN_PRINTF("No tokens to shift (n_past=%d, n_keep=%d)", n_past, n_keep);
    return success;
  }
  
  const int n_discard = chat_ctx->n_discard_tokens > 0 ? 
                        chat_ctx->n_discard_tokens : (n_left / 2);
  
  NN_INFO_PRINTF("Performing context shift: n_keep=%d, n_left=%d, n_discard=%d", 
                 n_keep, n_left, n_discard);
  
  // Perform the actual context shift using llama.cpp memory functions
  llama_memory_seq_rm(llama_get_memory(ctx), session_id, n_keep, n_keep + n_discard);
  llama_memory_seq_add(llama_get_memory(ctx), session_id, n_keep + n_discard, n_past, -n_discard);
  
  NN_INFO_PRINTF("Context shift completed successfully");
  return success;
}

// Partial KV cache deletion strategies
static wasi_nn_error clear_partial_kv_cache(LlamaChatContext* chat_ctx, uint32_t session_id, 
                                           const std::string& strategy) {
  if (!chat_ctx->enable_partial_cache_deletion) {
    NN_WARN_PRINTF("Partial cache deletion is disabled");
    return invalid_argument;
  }
  
  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;
  
  if (!ctx) {
    NN_ERR_PRINTF("No context available for cache deletion");
    return runtime_error;
  }
  
  const int n_ctx = llama_n_ctx(ctx);
  // Simplified approach - estimate current usage as 80% of context size
  const int n_past = n_ctx * 0.8f;
  
  if (strategy == "lru") {
    // Clear the oldest entries (simplified implementation)
    const int n_clear = n_past / 4; // Clear 25% of oldest entries
    
    if (n_clear > 0) {
      llama_memory_seq_rm(llama_get_memory(ctx), session_id, 0, n_clear);
      NN_INFO_PRINTF("Cleared %d oldest KV cache entries using LRU strategy", n_clear);
    }
  } else if (strategy == "fifo") {
    // Clear the newest entries
    const int n_clear = n_past / 4;
    
    if (n_clear > 0) {
      llama_memory_seq_rm(llama_get_memory(ctx), session_id, n_past - n_clear, n_past);
      NN_INFO_PRINTF("Cleared %d newest KV cache entries using FIFO strategy", n_clear);
    }
  } else if (strategy == "smart") {
    // Smart deletion based on token importance (simplified)
    const int n_keep = chat_ctx->n_keep_tokens;
    const int n_clear = (n_past - n_keep) / 2;
    
    if (n_clear > 0) {
      // Keep important tokens at the beginning and end, clear middle
      const int clear_start = n_keep + n_clear / 2;
      llama_memory_seq_rm(llama_get_memory(ctx), session_id, clear_start, clear_start + n_clear);
      NN_INFO_PRINTF("Cleared %d middle KV cache entries using smart strategy", n_clear);
    }
  } else {
    NN_ERR_PRINTF("Unknown cache deletion strategy: %s", strategy.c_str());
    return invalid_argument;
  }
  
  return success;
}

// Token cache reuse mechanism
static wasi_nn_error optimize_token_cache(LlamaChatContext* chat_ctx, uint32_t session_id) {
  if (!chat_ctx->enable_token_cache_reuse) {
    return success; // Not enabled, but not an error
  }
  
  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;
  
  if (!ctx) {
    NN_ERR_PRINTF("No context available for cache optimization");
    return runtime_error;
  }
  
  const int n_ctx = llama_n_ctx(ctx);
  // Simplified approach - estimate cached tokens
  const int n_cached = n_ctx * 0.7f; // Assume 70% cached
  
  if (n_cached > (int)chat_ctx->max_cache_tokens) {
    // Perform cache cleanup
    wasi_nn_error result = clear_partial_kv_cache(chat_ctx, session_id, 
                                                  chat_ctx->cache_deletion_strategy);
    if (result != success) {
      NN_WARN_PRINTF("Failed to optimize token cache: %d", result);
      return result;
    }
    
    chat_ctx->cache_hits++;
    NN_INFO_PRINTF("Token cache optimized: %d tokens cached, hit ratio: %.2f%%",
                    n_cached, 
                    (float)chat_ctx->cache_hits.load() / 
                    (chat_ctx->cache_hits.load() + chat_ctx->cache_misses.load()) * 100.0f);
  } else {
    chat_ctx->cache_misses++;
  }
  
  return success;
}

// Complete KV cache clear (based on server.cpp implementation)
static wasi_nn_error clear_kv_cache(LlamaChatContext* chat_ctx, uint32_t session_id) {
  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;
  
  if (!ctx) {
    NN_ERR_PRINTF("No context available for cache clearing");
    return runtime_error;
  }
  
  NN_INFO_PRINTF("Clearing KV cache for session %u", session_id);
  
  if (session_id == 0) {
    // Clear entire KV cache
    llama_memory_clear(llama_get_memory(ctx), true);
    NN_INFO_PRINTF("Cleared entire KV cache");
  } else {
    // Clear cache for specific session
    llama_memory_seq_rm(llama_get_memory(ctx), session_id, -1, -1);
    NN_INFO_PRINTF("Cleared KV cache for session %u", session_id);
  }
  
  return success;
}

// Memory pressure handling
static wasi_nn_error handle_memory_pressure(LlamaChatContext* chat_ctx) {
  NN_WARN_PRINTF("Memory pressure detected, initiating cleanup");
  
  // Strategy 1: Clear partial caches for all active sessions
  wasi_nn_error result = clear_partial_kv_cache(chat_ctx, 0, chat_ctx->cache_deletion_strategy);
  if (result != success) {
    NN_WARN_PRINTF("Partial cache cleanup failed, trying full cache clear");
    
    // Strategy 2: Clear entire cache if partial cleanup failed
    result = clear_kv_cache(chat_ctx, 0);
    if (result != success) {
      NN_ERR_PRINTF("Failed to handle memory pressure");
      return result;
    }
  }
  
  // Update memory tracking
  chat_ctx->current_memory_usage.store(get_current_memory_usage());
  
  NN_INFO_PRINTF("Memory pressure handling completed");
  return success;
}

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

// Phase 4.3: Parse advanced memory management configuration
static void parse_memory_config(const char *config_json, LlamaChatContext *chat_ctx)
{
  if (!config_json || !chat_ctx)
    return;

  cJSON *root = cJSON_Parse(config_json);
  if (!root)
  {
    NN_WARN_PRINTF("Failed to parse config JSON for memory settings");
    return;
  }

  cJSON *memory = cJSON_GetObjectItem(root, "memory");
  if (cJSON_IsObject(memory))
  {
    cJSON *item = nullptr;

    // Context shifting settings
    if ((item = cJSON_GetObjectItem(memory, "context_shifting")))
    {
      if (cJSON_IsBool(item))
      {
        chat_ctx->context_shifting_enabled = cJSON_IsTrue(item);
        NN_INFO_PRINTF("Context shifting %s", 
                       chat_ctx->context_shifting_enabled ? "enabled" : "disabled");
      }
    }

    // Cache strategy
    if ((item = cJSON_GetObjectItem(memory, "cache_strategy")))
    {
      if (cJSON_IsString(item))
      {
        chat_ctx->cache_strategy = cJSON_GetStringValue(item);
        NN_INFO_PRINTF("Cache strategy set to: %s", chat_ctx->cache_strategy.c_str());
      }
    }

    // Maximum cache tokens
    if ((item = cJSON_GetObjectItem(memory, "max_cache_tokens")))
    {
      if (cJSON_IsNumber(item))
      {
        chat_ctx->max_cache_tokens = (uint32_t)cJSON_GetNumberValue(item);
        NN_INFO_PRINTF("Max cache tokens set to: %u", chat_ctx->max_cache_tokens);
      }
    }

    // Phase 4.3: Advanced memory management settings
    if ((item = cJSON_GetObjectItem(memory, "n_keep_tokens")))
    {
      if (cJSON_IsNumber(item))
      {
        chat_ctx->n_keep_tokens = (uint32_t)cJSON_GetNumberValue(item);
        NN_INFO_PRINTF("Keep tokens set to: %u", chat_ctx->n_keep_tokens);
      }
    }

    if ((item = cJSON_GetObjectItem(memory, "n_discard_tokens")))
    {
      if (cJSON_IsNumber(item))
      {
        chat_ctx->n_discard_tokens = (uint32_t)cJSON_GetNumberValue(item);
        NN_INFO_PRINTF("Discard tokens set to: %u", chat_ctx->n_discard_tokens);
      }
    }

    if ((item = cJSON_GetObjectItem(memory, "memory_pressure_threshold")))
    {
      if (cJSON_IsNumber(item))
      {
        chat_ctx->memory_pressure_threshold = (float)cJSON_GetNumberValue(item);
        NN_INFO_PRINTF("Memory pressure threshold set to: %.2f", 
                       chat_ctx->memory_pressure_threshold);
      }
    }

    if ((item = cJSON_GetObjectItem(memory, "enable_partial_cache_deletion")))
    {
      if (cJSON_IsBool(item))
      {
        chat_ctx->enable_partial_cache_deletion = cJSON_IsTrue(item);
        NN_INFO_PRINTF("Partial cache deletion %s", 
                       chat_ctx->enable_partial_cache_deletion ? "enabled" : "disabled");
      }
    }

    if ((item = cJSON_GetObjectItem(memory, "enable_token_cache_reuse")))
    {
      if (cJSON_IsBool(item))
      {
        chat_ctx->enable_token_cache_reuse = cJSON_IsTrue(item);
        NN_INFO_PRINTF("Token cache reuse %s", 
                       chat_ctx->enable_token_cache_reuse ? "enabled" : "disabled");
      }
    }

    if ((item = cJSON_GetObjectItem(memory, "cache_deletion_strategy")))
    {
      if (cJSON_IsString(item))
      {
        chat_ctx->cache_deletion_strategy = cJSON_GetStringValue(item);
        NN_INFO_PRINTF("Cache deletion strategy set to: %s", 
                       chat_ctx->cache_deletion_strategy.c_str());
      }
    }

    if ((item = cJSON_GetObjectItem(memory, "max_memory_mb")))
    {
      if (cJSON_IsNumber(item))
      {
        chat_ctx->max_memory_mb = (uint32_t)cJSON_GetNumberValue(item);
        NN_INFO_PRINTF("Max memory limit set to: %u MB", chat_ctx->max_memory_mb);
      }
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

// ===============================================
// Phase 4.3: Forward declarations for internal memory management functions
// ===============================================
static wasi_nn_error auto_clear_kv_cache_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx);
static wasi_nn_error auto_clear_all_kv_cache(LlamaChatContext *chat_ctx);
static wasi_nn_error auto_perform_context_shift_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx);
static wasi_nn_error auto_optimize_memory(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx);

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
        
        // Enhanced task queue settings (Phase 4.2)
        cJSON *default_task_timeout = cJSON_GetObjectItem(config_obj, "default_task_timeout_ms");
        if (cJSON_IsNumber(default_task_timeout))
        {
          chat_ctx->default_task_timeout_ms = (uint32_t)default_task_timeout->valueint;
        }
        
        cJSON *priority_scheduling = cJSON_GetObjectItem(config_obj, "priority_scheduling_enabled");
        if (cJSON_IsBool(priority_scheduling))
        {
          chat_ctx->priority_scheduling_enabled = cJSON_IsTrue(priority_scheduling);
        }
        
        cJSON *fair_scheduling = cJSON_GetObjectItem(config_obj, "fair_scheduling_enabled");
        if (cJSON_IsBool(fair_scheduling))
        {
          chat_ctx->fair_scheduling_enabled = cJSON_IsTrue(fair_scheduling);
        }
        
        cJSON *queue_warning_threshold = cJSON_GetObjectItem(config_obj, "queue_warning_threshold");
        if (cJSON_IsNumber(queue_warning_threshold))
        {
          chat_ctx->queue_warning_threshold = (uint32_t)queue_warning_threshold->valueint;
        }
        
        cJSON *queue_reject_threshold = cJSON_GetObjectItem(config_obj, "queue_reject_threshold");
        if (cJSON_IsNumber(queue_reject_threshold))
        {
          chat_ctx->queue_reject_threshold = (uint32_t)queue_reject_threshold->valueint;
        }
        
        cJSON *auto_queue_cleanup = cJSON_GetObjectItem(config_obj, "auto_queue_cleanup");
        if (cJSON_IsBool(auto_queue_cleanup))
        {
          chat_ctx->auto_queue_cleanup = cJSON_IsTrue(auto_queue_cleanup);
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
        
        // New Phase 5.1 logging options
        cJSON *enable_timestamps = cJSON_GetObjectItem(logging, "timestamps");
        if (cJSON_IsBool(enable_timestamps))
        {
          chat_ctx->enable_timestamps = cJSON_IsTrue(enable_timestamps);
        }
        
        cJSON *enable_colors = cJSON_GetObjectItem(logging, "colors");
        if (cJSON_IsBool(enable_colors))
        {
          chat_ctx->enable_colors = cJSON_IsTrue(enable_colors);
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
    
    // Phase 4.3: Parse advanced memory management settings
    if (config && config_len > 0) {
      std::string config_str(config, config_len);
      parse_memory_config(config_str.c_str(), chat_ctx);
    }
  }

  // Initialize llama backend (from main.cpp)
  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  // Initialize task queue system (Phase 4.2)
  chat_ctx->task_queue = std::make_shared<wasi_nn_task_queue>();
  chat_ctx->task_queue->max_queue_size = chat_ctx->queue_size;
  
  // Start task processing thread if enabled
  if (chat_ctx->task_processing_enabled) {
    chat_ctx->task_processor_thread = std::thread([chat_ctx]() {
      NN_INFO_PRINTF("Task processor thread started");
      
      wasi_nn_task task;
      while (chat_ctx->task_queue->running) {
        if (chat_ctx->task_queue->dequeue_task(task, chat_ctx)) {
          // Process the task
          NN_INFO_PRINTF("Processing task %d for execution context %d", 
                         task.id, task.exec_ctx);
          
          // For now, just mark as completed
          // In a full implementation, this would trigger actual inference
          {
            std::unique_lock<std::mutex> lock(chat_ctx->task_queue->queue_mutex);
            chat_ctx->task_queue->tasks_completed++;
          }
          
          NN_INFO_PRINTF("Task %d completed", task.id);
        }
      }
      
      NN_INFO_PRINTF("Task processor thread terminated");
    });
  }

  NN_INFO_PRINTF("Llama chat backend initialized successfully");
  
  // Phase 5.1: Initialize advanced logging system
  initialize_advanced_logging(chat_ctx);
  
  // Use enhanced logging for configuration output
  WASI_NN_LOG_INFO(chat_ctx,
      "Session config: max_sessions=%d, idle_timeout_ms=%d, auto_cleanup=%s",
      chat_ctx->max_sessions, chat_ctx->idle_timeout_ms,
      chat_ctx->auto_cleanup_enabled ? "true" : "false");
  WASI_NN_LOG_INFO(chat_ctx,
      "Concurrency config: max_concurrent=%d, queue_size=%d",
      chat_ctx->max_concurrent, chat_ctx->queue_size);
  WASI_NN_LOG_INFO(chat_ctx,
      "Task Queue config: timeout=%dms, priority_scheduling=%s, fair_scheduling=%s",
      chat_ctx->default_task_timeout_ms,
      chat_ctx->priority_scheduling_enabled ? "true" : "false",
      chat_ctx->fair_scheduling_enabled ? "true" : "false");
  WASI_NN_LOG_INFO(chat_ctx,
      "Memory config: context_shifting=%s, cache_strategy=%s, max_cache_tokens=%d",
      chat_ctx->context_shifting_enabled ? "true" : "false",
      chat_ctx->cache_strategy.c_str(), chat_ctx->max_cache_tokens);
  WASI_NN_LOG_INFO(chat_ctx,
      "Logging config: level=%s, enable_debug=%s, timestamps=%s, colors=%s, file=%s",
      chat_ctx->log_level.c_str(),
      chat_ctx->enable_debug_log ? "true" : "false",
      chat_ctx->enable_timestamps ? "true" : "false",
      chat_ctx->enable_colors ? "true" : "false",
      chat_ctx->log_file.c_str());
  WASI_NN_LOG_INFO(chat_ctx,
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
    
    // Phase 4.3: Auto-clear KV cache for this session before closing
    auto_clear_kv_cache_session(chat_ctx, exec_ctx);
    
    chat_ctx->sessions.erase(it);
    if (chat_ctx->active_sessions > 0)
    {
      chat_ctx->active_sessions--; // Decrement active sessions counter
    }
    
    // Phase 4.3: Check if we should do global memory optimization after session close
    if (chat_ctx->active_sessions == 0) {
      // All sessions closed, good time for global cleanup
      auto_clear_all_kv_cache(chat_ctx);
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
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx || !wasi_nn_tensor)
    return invalid_argument;

  // Find the session
  auto session_it = chat_ctx->sessions.find(exec_ctx);
  if (session_it == chat_ctx->sessions.end())
    return invalid_argument;

  // Get the input prompt from tensor data
  const char *prompt_str = (const char *)wasi_nn_tensor->data;
  if (!prompt_str)
    return invalid_argument;

  // Calculate tensor size from dimensions
  uint32_t tensor_size = 1;
  if (wasi_nn_tensor->dimensions && wasi_nn_tensor->dimensions->buf) {
    for (uint32_t i = 0; i < wasi_nn_tensor->dimensions->size; ++i) {
      tensor_size *= wasi_nn_tensor->dimensions->buf[i];
    }
  }
  
  // For text data, we assume it's null-terminated or use a reasonable max length
  size_t prompt_len = strnlen(prompt_str, tensor_size);
  std::string prompt(prompt_str, prompt_len);
  
  // Store the prompt for later processing in compute()
  // For now, we'll store it in the session (could be optimized)
  session_it->second.session_id = prompt; // Temporary storage
  
  NN_INFO_PRINTF("Input set for execution context %d: %.100s%s", 
                 exec_ctx, prompt.c_str(), 
                 prompt.length() > 100 ? "..." : "");

  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
compute(void *ctx, graph_execution_context exec_ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  // Phase 4.3: Automatic memory optimization before processing
  wasi_nn_error opt_result = auto_optimize_memory(chat_ctx, exec_ctx);
  if (opt_result != success) {
    NN_WARN_PRINTF("Memory optimization warning for session %u: %d", exec_ctx, opt_result);
    // Continue with inference even if optimization has issues
  }

  // Find the session
  auto session_it = chat_ctx->sessions.find(exec_ctx);
  if (session_it == chat_ctx->sessions.end())
    return invalid_argument;

  // Check if we're at capacity - if so, queue the task
  if (chat_ctx->active_sessions >= chat_ctx->max_concurrent)
  {
    if (!chat_ctx->task_queue)
    {
      NN_WARN_PRINTF("Task queue not initialized but needed for queuing");
      return runtime_error;
    }
    
    // Create a task for queuing
    wasi_nn_task task;
    task.exec_ctx = exec_ctx;
    task.prompt = session_it->second.session_id; // Retrieved from set_input
    task.timeout_ms = chat_ctx->default_task_timeout_ms;
    task.timeout_at = task.created_at + std::chrono::milliseconds(task.timeout_ms);
    task.is_queued = true;
    
    // Determine priority (for now, all tasks are normal priority)
    // In a real implementation, this could be based on user settings or prompt analysis
    task.priority = WASI_NN_PRIORITY_NORMAL;
    
    // Try to enqueue the task
    if (!chat_ctx->task_queue->enqueue_task(std::move(task), chat_ctx))
    {
      NN_WARN_PRINTF("Failed to enqueue task for execution context %d - queue full", exec_ctx);
      return runtime_error;
    }
    
    NN_INFO_PRINTF("Task queued for execution context %d due to capacity limits (%d/%d active)", 
                   exec_ctx, chat_ctx->active_sessions, chat_ctx->max_concurrent);
    return success;
  }
  
  // If we have capacity, process immediately
  NN_INFO_PRINTF("Processing compute request immediately for execution context %d", exec_ctx);
  
  // Update last activity time
  session_it->second.last_activity = std::chrono::steady_clock::now();
  
  // Phase 4.3: Auto context shift if needed (context window approaching limit)
  auto_perform_context_shift_session(chat_ctx, exec_ctx);
  
  // For Phase 4.2, we're mainly implementing the queuing mechanism
  // The actual inference processing remains the same as before
  // This would typically involve:
  // 1. Creating server tasks
  // 2. Processing through the server context
  // 3. Managing the llama context and sampling
  
  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
get_output(void *ctx, graph_execution_context exec_ctx, uint32_t index,
           tensor_data output_tensor, uint32_t *output_tensor_size)
{
  return success;
}

// Phase 4.3: Internal Memory Management Functions
// ===============================================
// These functions are automatically called during inference for optimization

static wasi_nn_error
auto_clear_kv_cache_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }
  
  NN_DBG_PRINTF("Auto-clearing KV cache for session %u", exec_ctx);
  
  wasi_nn_error result = clear_kv_cache(chat_ctx, exec_ctx);
  if (result != success) {
    NN_WARN_PRINTF("Failed to auto-clear KV cache for session %u: %d", exec_ctx, result);
    return result;
  }
  
  return success;
}

static wasi_nn_error
auto_clear_all_kv_cache(LlamaChatContext *chat_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }
  
  NN_DBG_PRINTF("Auto-clearing all KV cache");
  
  wasi_nn_error result = clear_kv_cache(chat_ctx, 0); // session_id = 0 means all sessions
  if (result != success) {
    NN_WARN_PRINTF("Failed to auto-clear all KV cache: %d", result);
    return result;
  }
  
  return success;
}

static wasi_nn_error
auto_perform_context_shift_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }

  if (!chat_ctx->context_shifting_enabled) {
    NN_DBG_PRINTF("Context shifting is disabled for session %u", exec_ctx);
    return success; // Not an error, just disabled
  }
  
  NN_DBG_PRINTF("Auto-performing context shift for session %u", exec_ctx);
  
  wasi_nn_error result = perform_context_shift(chat_ctx, exec_ctx);
  if (result != success) {
    NN_WARN_PRINTF("Failed to auto-perform context shift for session %u: %d", exec_ctx, result);
    return result;
  }
  
  return success;
}

static wasi_nn_error
auto_optimize_memory(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }
  
  NN_DBG_PRINTF("Auto-optimizing memory for session %u", exec_ctx);
  
  // Check for memory pressure and handle it
  if (check_memory_pressure(chat_ctx)) {
    NN_INFO_PRINTF("Memory pressure detected, performing automatic cleanup");
    wasi_nn_error result = handle_memory_pressure(chat_ctx);
    if (result != success) {
      NN_WARN_PRINTF("Failed to handle memory pressure: %d", result);
      // Don't fail the inference, just log warning
    }
  }
  
  // Optimize token cache (non-critical)
  wasi_nn_error result = optimize_token_cache(chat_ctx, exec_ctx);
  if (result != success) {
    NN_DBG_PRINTF("Token cache optimization skipped: %d", result);
    // This is not critical for inference
  }
  
  return success;
}
