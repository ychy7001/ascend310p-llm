// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
#ifdef PROBE_FP16_CONTROL
using ProbeB = half;
#else
using ProbeB = int8_t;
#endif

// Compile-only capability probe, deliberately isolated from the working C8
// library. This tests Mmad's input type support, not a valid end-to-end layout.
// Never launch this kernel: it does not initialize its input tensors.
extern "C" __global__ __aicore__ void c8_mixed_mmad_probe_kernel(GM_ADDR output) {
  TPipe pipe;
  TBuf<TPosition::A2> ab;
  TBuf<TPosition::B2> bb;
  TBuf<TPosition::CO1> cb;
  pipe.InitBuffer(ab, 16 * 256 * sizeof(half));
  pipe.InitBuffer(bb, 16 * 256 * sizeof(ProbeB));
  pipe.InitBuffer(cb, 16 * 16 * sizeof(float));
  auto a = ab.Get<half>();
  auto b = bb.Get<ProbeB>();
  auto c = cb.Get<float>();
  MmadParams params;
  params.m = 16;
  params.n = 16;
  params.k = 256;
  params.cmatrixInitVal = true;
  Mmad(c, a, b, params);
  pipe_barrier(PIPE_ALL);
}
