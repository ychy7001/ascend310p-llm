// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Resolve one logical page from device metadata. Output is at most one page;
// no host scalar read or full-context FP16 materialization is required.
extern "C" __global__ __aicore__ void c8_table_dequant_kernel(
    GM_ADDR input, GM_ADDR scale, GM_ADDR table, GM_ADDR output,
    int64_t logical_page, int64_t block_size, int64_t num_blocks,
    int64_t tokens, int64_t heads, int64_t dim) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> ib, sb, ob, fb;
  pipe.InitBuffer(ib, 32768 * sizeof(int8_t));
  pipe.InitBuffer(sb, 16 * sizeof(half));
  pipe.InitBuffer(ob, 32768 * sizeof(half));
  pipe.InitBuffer(fb, 32768 * sizeof(half));
  GlobalTensor<int8_t> src;
  GlobalTensor<int32_t> mapping;
  GlobalTensor<half> scales, dst;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(input));
  mapping.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
  scales.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  auto xi = ib.Get<int8_t>();
  auto sh = sb.Get<half>();
  auto yh = ob.Get<half>();
  auto xf = fb.Get<half>();
  int64_t width = heads * dim;
  int64_t t = GetBlockIdx();
  if (t >= tokens) return;
  int64_t physical = mapping.GetValue(logical_page);
  if (physical < 0 || physical >= num_blocks) {
    // Padding or invalid mapping must never access outside the cache.
    Duplicate(yh, static_cast<half>(0), width);
  } else {
    int64_t slot = physical * block_size + t;
    DataCopy(xi, src[slot * width], width);
    DataCopy(sh, scales[slot * 16], 16);
    auto ready = pipe.FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(ready);
    WaitFlag<HardEvent::MTE2_V>(ready);
    Cast(xf, xi, RoundMode::CAST_NONE, width);
    pipe_barrier(PIPE_V);
    for (int64_t h = 0; h < heads; ++h) {
      Duplicate(yh[h * dim], sh.GetValue(h), dim);
    }
    pipe_barrier(PIPE_V);
    Mul(yh, xf, yh, width);
  }
  pipe_barrier(PIPE_V);
  auto write = pipe.FetchEventID(HardEvent::V_MTE3);
  SetFlag<HardEvent::V_MTE3>(write);
  WaitFlag<HardEvent::V_MTE3>(write);
  DataCopy(dst[t * width], yh, width);
  auto done = pipe.FetchEventID(HardEvent::MTE3_V);
  SetFlag<HardEvent::MTE3_V>(done);
  WaitFlag<HardEvent::MTE3_V>(done);
}
