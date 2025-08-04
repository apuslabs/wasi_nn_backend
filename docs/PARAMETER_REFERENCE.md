# WASI-NN Backend Parameter Reference

## Table of Contents

1. [Backend Configuration](#backend-configuration)
2. [Model Parameters](#model-parameters)
3. [Sampling Parameters](#sampling-parameters)
4. [Memory Management](#memory-management)
5. [Logging Configuration](#logging-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Advanced Features](#advanced-features)
8. [Platform-Specific Settings](#platform-specific-settings)

## Configuration File Structure

The configuration is provided as JSON with the following top-level sections:

```json
{
  "backend": { /* Backend management settings */ },
  "model": { /* Model loading and context settings */ },
  "sampling": { /* Text generation parameters */ },
  "stopping": { /* Generation stopping criteria */ },
  "memory": { /* Memory and cache management */ },
  "logging": { /* Logging and debugging */ },
  "performance": { /* Performance optimization */ },
  "logit_bias": [ /* Token bias adjustments */ ]
}
```

## Backend Configuration

Controls the overall behavior of the WASI-NN backend system.

### Session Management

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `max_sessions` | integer | 100 | 1-10000 | Maximum number of concurrent sessions | 最大并发会话数 |
| `idle_timeout_ms` | integer | 300000 | 1000-86400000 | Session idle timeout in milliseconds | 会话空闲超时（毫秒） |
| `auto_cleanup` | boolean | true | - | Enable automatic cleanup of idle sessions | 启用空闲会话的自动清理 |
| `max_concurrent` | integer | 10 | 1-256 | Maximum concurrent active operations | 最大并发活动操作数 |

**Example:**
```json
{
  "backend": {
    "max_sessions": 200,
    "idle_timeout_ms": 600000,
    "auto_cleanup": true,
    "max_concurrent": 20
  }
}
```

### Task Queue Management

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `queue_size` | integer | 500 | 1-10000 | Maximum task queue size | 最大任务队列大小 |
| `default_task_timeout_ms` | integer | 30000 | 1000-600000 | Default task timeout in milliseconds | 默认任务超时（毫秒） |
| `priority_scheduling_enabled` | boolean | true | - | Enable priority-based task scheduling | 启用基于优先级的任务调度 |
| `fair_scheduling_enabled` | boolean | true | - | Enable fair scheduling across users | 启用用户间的公平调度 |
| `auto_queue_cleanup` | boolean | true | - | Automatically cleanup expired tasks | 自动清理过期任务 |
| `queue_warning_threshold` | integer | 400 | 1-queue_size | Queue size warning threshold | 队列大小警告阈值 |
| `queue_reject_threshold` | integer | 500 | 1-queue_size | Queue size rejection threshold | 队列大小拒绝阈值 |

**Example:**
```json
{
  "backend": {
    "queue_size": 1000,
    "default_task_timeout_ms": 45000,
    "priority_scheduling_enabled": true,
    "fair_scheduling_enabled": true,
    "queue_warning_threshold": 800,
    "queue_reject_threshold": 1000
  }
}
```

## Model Parameters

Controls model loading, context management, and basic inference settings.

### Core Model Settings

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `n_ctx` | integer | 2048 | 128-32768 | Context window size in tokens | 上下文窗口大小（令牌数） |
| `ctx_size` | integer | 2048 | 128-32768 | Alias for n_ctx | n_ctx 的别名 |
| `n_batch` | integer | 512 | 1-2048 | Batch size for prompt processing | 提示处理的批处理大小 |
| `batch_size` | integer | 512 | 1-2048 | Alias for n_batch | n_batch 的别名 |
| `n_gpu_layers` | integer | 0 | 0-999 | Number of layers to offload to GPU | 卸载到 GPU 的层数 |
| `threads` | integer | 8 | 1-64 | Number of CPU threads to use | 使用的 CPU 线程数 |

**Recommendations:**
- **Small models (< 7B parameters)**: `n_ctx: 4096, n_batch: 512`
- **Large models (> 13B parameters)**: `n_ctx: 2048, n_batch: 256`

**Example:**
```json
{
  "model": {
    "n_ctx": 4096,
    "n_batch": 512,
    "n_gpu_layers": 32,
    "threads": 12
  }
}
```

### Hardware Optimization

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `use_mmap` | boolean | true | - | Use memory mapping for model loading | 使用内存映射加载模型 |
| `use_mlock` | boolean | false | - | Lock model in physical memory | 将模型锁定在物理内存中 |
| `numa` | string | "disabled" | disabled/distribute/isolate/numactl | NUMA memory strategy | NUMA 内存策略 |

**NUMA Strategies:**
- `disabled`: No NUMA optimization
- `distribute`: Distribute memory across NUMA nodes
- `isolate`: Isolate to specific NUMA node
- `numactl`: Use numactl for advanced control

## Sampling Parameters

Controls text generation quality and behavior. These parameters significantly affect output quality and creativity.

### Core Sampling

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `temperature` | float | 0.7 | 0.0-2.0 | Randomness in token selection (higher = more random) | 令牌选择的随机性（越高越随机） |
| `temp` | float | 0.7 | 0.0-2.0 | Alias for temperature | temperature 的别名 |
| `top_p` | float | 0.9 | 0.0-1.0 | Nucleus sampling threshold | 核采样阈值 |
| `top_k` | integer | 40 | -1-200 | Top-k sampling limit (-1 = disabled) | Top-k 采样限制（-1 = 禁用） |
| `min_p` | float | 0.05 | 0.0-1.0 | Minimum probability threshold | 最小概率阈值 |
| `typical_p` | float | 1.0 | 0.0-1.0 | Typical sampling parameter | 典型采样参数 |

**Temperature Guidelines:**
- `0.1-0.3`: Very focused, deterministic
- `0.4-0.7`: Balanced creativity and coherence
- `0.8-1.2`: More creative and diverse
- `1.3-2.0`: Highly random, experimental

**Example:**
```json
{
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "typical_p": 1.0
  }
}
```

### Penalty Parameters

Control repetition and encourage diversity in generated text.

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `repeat_penalty` | float | 1.1 | 0.0-2.0 | Penalty for repeating tokens | 重复令牌的惩罚 |
| `presence_penalty` | float | 0.0 | -2.0-2.0 | Penalty for token presence | 令牌存在的惩罚 |
| `frequency_penalty` | float | 0.0 | -2.0-2.0 | Penalty based on token frequency | 基于令牌频率的惩罚 |
| `penalty_last_n` | integer | 64 | -1-2048 | Tokens to consider for penalty (-1 = ctx_size) | 考虑惩罚的令牌数（-1 = 上下文大小） |
| `repeat_last_n` | integer | 64 | -1-2048 | Alias for penalty_last_n | penalty_last_n 的别名 |

**Penalty Guidelines:**
- `repeat_penalty`: 1.0 = no penalty, >1.0 = discourage repetition, <1.0 = encourage repetition
- `presence_penalty`: Positive values discourage using tokens that have appeared
  `presence_penalty`: 正值阻止使用已出现的令牌
- `frequency_penalty`: Positive values penalize frequently used tokens
  `frequency_penalty`: 正值惩罚频繁使用的令牌

### Advanced Sampling

#### DRY Sampling (Discourage Repetition Sampling)

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `dry_multiplier` | float | 0.0 | 0.0-5.0 | DRY sampling strength (0.0 = disabled) | DRY 采样强度（0.0 = 禁用） |
| `dry_base` | float | 1.75 | 1.0-4.0 | DRY base value | DRY 基础值 |
| `dry_allowed_length` | integer | 2 | 1-20 | Minimum sequence length to apply DRY | 应用 DRY 的最小序列长度 |
| `dry_penalty_last_n` | integer | -1 | -1-2048 | Tokens to consider for DRY (-1 = ctx_size) | DRY 考虑的令牌数（-1 = 上下文大小） |
| `dry_sequence_breakers` | array | ["\n", ":", "\"", "*"] | - | Sequences that break DRY penalties | 打破 DRY 惩罚的序列 |

**DRY Sampling Usage:**
```json
{
  "sampling": {
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_sequence_breakers": ["\n", ":", "\"", "*", "```"]
  }
}
```

#### Dynamic Temperature

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `dynatemp_range` | float | 0.0 | 0.0-2.0 | Dynamic temperature range (0.0 = disabled) | 动态温度范围（0.0 = 禁用） |
| `dynatemp_exponent` | float | 1.0 | 0.1-5.0 | Dynamic temperature exponent | 动态温度指数 |

**Dynamic Temperature Explanation:**
Dynamic temperature adjusts the temperature based on the entropy of the prediction, providing more randomness for uncertain predictions and less for confident ones.

#### Mirostat Sampling

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `mirostat` | integer | 0 | 0-2 | Mirostat algorithm version (0 = disabled) | Mirostat 算法版本（0 = 禁用） |
| `mirostat_tau` | float | 5.0 | 0.1-10.0 | Target entropy (perplexity) | 目标熵（困惑度） |
| `mirostat_eta` | float | 0.1 | 0.001-1.0 | Learning rate for entropy control | 熵控制的学习率 |

**Mirostat Guidelines:**
- Version 1: Original Mirostat algorithm
- Version 2: Improved version with better stability
- `tau`: Lower values = more focused, higher values = more diverse

### Other Sampling Options

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `seed` | integer | -1 | -1-2³¹ | Random seed (-1 = random) | 随机种子（-1 = 随机） |
| `n_probs` | integer | 0 | 0-100 | Number of top probabilities to return | 返回的顶部概率数量 |
| `logprobs` | integer | 0 | 0-100 | Alias for n_probs (OpenAI compatibility) | n_probs 的别名（OpenAI 兼容） |
| `min_keep` | integer | 1 | 1-100 | Minimum tokens to keep in sampling | 采样中保留的最小令牌数 |
| `ignore_eos` | boolean | false | - | Ignore end-of-sequence tokens | 忽略序列结束令牌 |

## Stopping Criteria

Controls when text generation should stop.

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `max_tokens` | integer | 512 | 1-4096 | Maximum tokens to generate | 生成的最大令牌数 |
| `n_predict` | integer | 512 | 1-4096 | Alias for max_tokens | max_tokens 的别名 |
| `stop` | array | [] | - | Stop sequences (strings that end generation) | 停止序列（结束生成的字符串） |
| `ignore_eos` | boolean | false | - | Ignore model's end-of-sequence token | 忽略模型的序列结束令牌 |

**Example:**
```json
{
  "stopping": {
    "max_tokens": 1024,
    "stop": ["</s>", "[INST]", "[/INST]", "Human:", "Assistant:"],
    "ignore_eos": false
  }
}
```

## Memory Management

Advanced memory management features for production deployments.

### Context Management

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `context_shifting` | boolean | true | - | Enable automatic context shifting | 启用自动上下文切换 |
| `n_keep_tokens` | integer | 128 | 64-2048 | Tokens to keep during context shift | 上下文切换时保留的令牌数 |
| `n_discard_tokens` | integer | 256 | 128-1024 | Tokens to discard during shift | 切换时丢弃的令牌数 |

### Cache Management

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `cache_strategy` | string | "lru" | lru/fifo/smart | Cache replacement strategy | 缓存替换策略 |
| `max_cache_tokens` | integer | 100000 | 1024-1000000 | Maximum cached tokens | 最大缓存令牌数 |
| `enable_partial_cache_deletion` | boolean | true | - | Allow partial cache clearing | 允许部分缓存清除 |
| `enable_token_cache_reuse` | boolean | true | - | Reuse cached tokens across sessions | 跨会话重用缓存令牌 |
| `cache_deletion_strategy` | string | "lru" | lru/fifo/smart | Strategy for cache deletion | 缓存删除策略 |

**Cache Strategies:**
- `lru`: Least Recently Used
- `fifo`: First In, First Out
- `smart`: Adaptive strategy based on usage patterns

### Memory Limits

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `max_memory_mb` | integer | 8192 | 0-32768 | Maximum memory usage in MB (0 = unlimited) | 最大内存使用量（MB）（0 = 无限制） |
| `memory_pressure_threshold` | float | 0.8 | 0.5-0.95 | Memory pressure threshold (0.8 = 80%) | 内存压力阈值（0.8 = 80%） |

**Example:**
```json
{
  "memory": {
    "context_shifting": true,
    "cache_strategy": "lru",
    "max_cache_tokens": 200000,
    "memory_pressure_threshold": 0.85,
    "n_keep_tokens": 256,
    "enable_partial_cache_deletion": true,
    "max_memory_mb": 16384
  }
}
```

## Logging Configuration

Comprehensive logging system for debugging and monitoring.

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `level` | string | "info" | debug/info/warn/error/fatal | Minimum log level | 最小日志级别 |
| `enable_debug` | boolean | false | - | Enable debug-level logging | 启用调试级别日志 |
| `timestamps` | boolean | true | - | Include timestamps in logs | 在日志中包含时间戳 |
| `colors` | boolean | true | - | Enable colored output | 启用彩色输出 |
| `file` | string | "" | - | Log file path (empty = stdout only) | 日志文件路径（空 = 仅标准输出） |

**Log Levels:**
- `debug`: Detailed debugging information
- `info`: General information messages
- `warn`: Warning messages
- `error`: Error messages
- `fatal`: Fatal error messages

**Example:**
```json
{
  "logging": {
    "level": "info",
    "enable_debug": false,
    "timestamps": true,
    "colors": true,
    "file": "/var/log/wasi_nn_backend.log"
  }
}
```

## Performance Tuning

Optimization settings for different deployment scenarios.

### Batch Processing

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `batch_processing` | boolean | true | - | Enable batch processing of requests | 启用请求的批处理 |
| `batch_size` | integer | 512 | 1-2048 | Processing batch size | 处理批处理大小 |
| `batch_timeout_ms` | integer | 100 | 10-1000 | Maximum wait time for batch completion | 批处理完成的最大等待时间 |

## Advanced Features

### Grammar and Constraints

| Parameter | Type | Default | Range | Description (EN) | Description (CN) |
|-----------|------|---------|--------|------------------|------------------|
| `grammar` | string | "" | - | GBNF grammar for structured output | 用于结构化输出的 GBNF 语法 |
| `grammar_lazy` | boolean | false | - | Enable lazy grammar evaluation | 启用延迟语法评估 |

**Grammar Example:**
```json
{
  "sampling": {
    "grammar": "root ::= object\nobject ::= \"{\" ws member (ws \",\" ws member)* ws \"}\"\nmember ::= string ws \":\" ws value\nstring ::= \"\\\"\" char* \"\\\"\"\nvalue ::= string | number | \"true\" | \"false\" | \"null\"\nchar ::= [^\"\\\\] | \"\\\\\" escape\nescape ::= [\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]\nnumber ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [\"+\"-]? [0-9]+)?\nws ::= [ \\t\\n]*"
  }
}
```

### Logit Bias

Fine-tune token probabilities for specific use cases.

```json
{
  "logit_bias": [
    [198, -1.0],    // Reduce probability of newline token
    [628, 0.5],     // Increase probability of specific token
    [50256, -100]   // Effectively ban a token
  ]
}
```

Each entry is `[token_id, bias_value]` where:
- Positive bias increases probability
- Negative bias decreases probability
- Large negative values (-100) effectively ban tokens

## Platform-Specific Settings

### Linux Optimization

```json
{
  "model": {
    "numa": "distribute",
    "use_mlock": true,
    "threads": 16
  },
  "performance": {
    "batch_processing": true,
    "batch_size": 1024
  }
}
```

### macOS Optimization

```json
{
  "model": {
    "n_gpu_layers": 1,
    "threads": 8,
    "use_mmap": true
  }
}
```

### Windows Optimization

```json
{
  "model": {
    "threads": 12,
    "use_mmap": false,
    "n_gpu_layers": 32
  }
}
```

## Configuration Examples

### Production Server

High-throughput server configuration for production environments.

```json
{
  "backend": {
    "max_sessions": 500,
    "idle_timeout_ms": 600000,
    "auto_cleanup": true,
    "max_concurrent": 50,
    "queue_size": 2000,
    "priority_scheduling_enabled": true,
    "fair_scheduling_enabled": true
  },
  "model": {
    "n_ctx": 4096,
    "n_batch": 512,
    "n_gpu_layers": 40,
    "threads": 16,
    "use_mmap": true,
    "use_mlock": true
  },
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "dry_multiplier": 0.8,
    "dry_base": 1.75
  },
  "memory": {
    "context_shifting": true,
    "cache_strategy": "smart",
    "max_cache_tokens": 500000,
    "memory_pressure_threshold": 0.85,
    "max_memory_mb": 32768
  },
  "logging": {
    "level": "info",
    "timestamps": true,
    "file": "/var/log/wasi_nn_backend.log"
  },
  "performance": {
    "batch_processing": true,
    "batch_size": 1024
  }
}
```

### Development Environment

Configuration optimized for development and testing.

```json
{
  "backend": {
    "max_sessions": 10,
    "idle_timeout_ms": 60000,
    "auto_cleanup": true,
    "max_concurrent": 5
  },
  "model": {
    "n_ctx": 2048,
    "n_batch": 256,
    "n_gpu_layers": 0,
    "threads": 4
  },
  "sampling": {
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": 42
  },
  "logging": {
    "level": "debug",
    "enable_debug": true,
    "timestamps": true,
    "colors": true
  }
}
```

### Low-Resource Environment

Configuration for systems with limited resources.

```json
{
  "backend": {
    "max_sessions": 5,
    "max_concurrent": 2,
    "queue_size": 50
  },
  "model": {
    "n_ctx": 1024,
    "n_batch": 128,
    "n_gpu_layers": 0,
    "threads": 2
  },
  "memory": {
    "max_cache_tokens": 10000,
    "memory_pressure_threshold": 0.9,
    "max_memory_mb": 2048
  }
}
```

## Validation Rules

The backend validates configuration parameters and will use defaults or report errors for invalid values:

### Automatic Adjustments

- `penalty_last_n = -1` → automatically set to `n_ctx`
- `dry_penalty_last_n = -1` → automatically set to `n_ctx`
- Queue thresholds adjusted to not exceed `queue_size`

### Error Conditions

- `temperature < 0` or `temperature > 2.0` → Warning and reset to default
- `n_ctx < 128` → Error, minimum context size required
- `max_memory_mb < 64` (unless 0) → Warning and reset to default

This comprehensive parameter reference provides detailed information for fine-tuning the WASI-NN backend for your specific use case, whether it's a high-performance production server or a resource-constrained development environment.
