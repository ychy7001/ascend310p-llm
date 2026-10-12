// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Copy one quantized token and its 16-half aligned scale row into a physical
// cache slot. One block handles one logical token, matching page dequant.
extern "C" __global__ __aicore__ void c8_page_scatter_kernel(
    GM_ADDR input, GM_ADDR scale, GM_ADDR slots, GM_ADDR cache,
    GM_ADDR scale_cache, int64_t tokens, int64_t heads,
    int64_t scale_stride, int64_t width, int64_t capacity) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> ib, sb;
  pipe.InitBuffer(ib, 32768 * sizeof(int8_t));
  pipe.InitBuffer(sb, 16 * sizeof(half));
  GlobalTensor<int8_t> src, dst;
  GlobalTensor<half> scales, dst_scales;
  GlobalTensor<int32_t> slot_tensor;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(input));
  scales.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(cache));
  dst_scales.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale_cache));
  slot_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slots));
  auto xi = ib.Get<int8_t>();
  auto sh = sb.Get<half>();
  int64_t token = GetBlockIdx();
  if (token >= tokens) return;
  int32_t slot = slot_tensor.GetValue(token);
  // vLLM uses -1 for padded token slots. Reject other invalid slots without
  // issuing an out-of-bounds GM access; callers validate positive slot ranges.
  if (slot < 0 || slot >= capacity) return;
  DataCopy(xi, src[token * width], width);
  DataCopy(sh, scales[token * scale_stride], scale_stride);
  auto ready = pipe.FetchEventID(HardEvent::MTE2_MTE3);
  SetFlag<HardEvent::MTE2_MTE3>(ready);
  WaitFlag<HardEvent::MTE2_MTE3>(ready);
  DataCopy(dst[static_cast<int64_t>(slot) * width], xi, width);
  DataCopy(dst_scales[static_cast<int64_t>(slot) * scale_stride], sh, scale_stride);
}
