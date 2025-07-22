/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef LLAMA_CHAT_LIB_H
#define LLAMA_CHAT_LIB_H

#include "../include/wasi_nn_llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Main API functions (matching wasi_nn interface)
wasi_nn_error init_backend(void **ctx);
wasi_nn_error init_backend_with_config(void **ctx, const char *config,
                                       uint32_t config_len);
wasi_nn_error deinit_backend(void *ctx);
wasi_nn_error load(void *ctx, graph_builder_array *builder,
                   graph_encoding encoding, execution_target target, graph *g);
wasi_nn_error load_by_name(void *ctx, const char *filename,
                           uint32_t filename_len, graph *g);
wasi_nn_error load_by_name_with_config(void *ctx, const char *filename,
                                       uint32_t filename_len,
                                       const char *config, uint32_t config_len,
                                       graph *g);
wasi_nn_error init_execution_context(void *ctx, const char *session_id,
                                     graph_execution_context *exec_ctx);
wasi_nn_error close_execution_context(void *ctx,
                                      graph_execution_context exec_ctx);
wasi_nn_error set_input(void *ctx, graph_execution_context exec_ctx,
                        uint32_t index, tensor *wasi_nn_tensor);
wasi_nn_error compute(void *ctx, graph_execution_context exec_ctx);
wasi_nn_error get_output(void *ctx, graph_execution_context exec_ctx,
                         uint32_t index, tensor_data output_tensor,
                         uint32_t *output_tensor_size);
wasi_nn_error run_inference(void *ctx, graph_execution_context exec_ctx,
                            uint32_t index, tensor *input_tensor,
                            tensor_data output_tensor,
                            uint32_t *output_tensor_size, const char *options);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_CHAT_LIB_H