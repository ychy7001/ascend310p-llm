// SPDX-License-Identifier: Apache-2.0
// Minimal independent namespace for the 310P W4A8 operators.
// It avoids loading a second vllm_ascend_C, whose _C_ascend namespace conflicts
// with the official vLLM-Ascend extension.
#include <torch/library.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "gmm/qwen_w4_a8_pack_v310/qwen_w4_a8_pack_310_torch_adpt.h"
#include "gmm/qwen_w4_a8_swiglu_pack_v310/qwen_w4_a8_swiglu_pack_310_torch_adpt.h"
#include "gmm/qwen_w4_a8_int4_matmul_v310/qwen_w4_a8_int4_matmul_310_torch_adpt.h"
#include "gmm/qwen_w4_a8_int4_down_reduce_v310/qwen_w4_a8_int4_down_reduce_310_torch_adpt.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

TORCH_LIBRARY(qwen36_int4, m) {
  m.def("pack(Tensor x) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("swiglu_pack(Tensor gate_up) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("matmul(Tensor low, Tensor high, Tensor activation_scale, Tensor activation_sum, Tensor codes, Tensor scale, Tensor offset, Tensor weight_sum, Tensor group_ends) -> Tensor");
  m.def("down_reduce(Tensor low, Tensor high, Tensor activation_scale, Tensor activation_sum, Tensor codes, Tensor scale, Tensor offset, Tensor weight_sum, Tensor route_ids, Tensor route_weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(qwen36_int4, PrivateUse1, m) {
  m.impl("pack", &vllm_ascend::npu_qwen_w4_a8_pack_310);
  m.impl("swiglu_pack", &vllm_ascend::npu_qwen_w4_a8_swiglu_pack_310);
  m.impl("matmul", &vllm_ascend::npu_qwen_w4_a8_int4_matmul_310);
  m.impl("down_reduce", &vllm_ascend::npu_qwen_w4_a8_int4_down_reduce_310);
}
