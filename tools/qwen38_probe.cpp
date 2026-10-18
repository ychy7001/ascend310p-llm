// SPDX-License-Identifier: Apache-2.0
// Probe uses pinned community Torch adapters, plus our existing independent W4 namespace.
#include "qwen36_int4_custom.cpp"
#include "attention/qsa_sparse_attention_v310/qsa_sparse_attention_310_torch_adpt.h"
TORCH_LIBRARY(qwen38_probe, m) {
  m.def("qsa(Tensor query, Tensor key_cache, Tensor value_cache, Tensor group_indices, Tensor group_counts, Tensor tail_starts, Tensor tail_counts, Tensor block_table, Tensor query_start_loc, float scale, int compress_ratio, int logical_kv_heads) -> Tensor");
}
TORCH_LIBRARY_IMPL(qwen38_probe, PrivateUse1, m) {
  m.impl("qsa", &vllm_ascend::npu_qsa_sparse_attention_310);
}
