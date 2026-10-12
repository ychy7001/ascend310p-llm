// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Isolated 16x16 QK tile probe for 310P's basic Cube path. This establishes
// the packing and L0C readback before inserting Cube into fused C8 attention.
extern "C" __global__ __aicore__ void c8_qk_cube_kernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR output, int64_t dim, bool transpose_b) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> qu, ku, cu;
  TBuf<TPosition::A1> a1b;
  TBuf<TPosition::B1> b1b;
  TBuf<TPosition::A2> a2b;
  TBuf<TPosition::B2> b2b;
  TBuf<TPosition::CO1> c1b;
  pipe.InitBuffer(qu, 16 * 256 * sizeof(half));
  pipe.InitBuffer(ku, 16 * 256 * sizeof(half));
  pipe.InitBuffer(cu, 16 * 16 * sizeof(float));
  pipe.InitBuffer(a1b, 16 * 256 * sizeof(half));
  pipe.InitBuffer(b1b, 16 * 256 * sizeof(half));
  pipe.InitBuffer(a2b, 16 * 256 * sizeof(half));
  pipe.InitBuffer(b2b, 16 * 256 * sizeof(half));
  pipe.InitBuffer(c1b, 16 * 16 * sizeof(float));
  auto q=qu.Get<half>(); auto k=ku.Get<half>(); auto c=cu.Get<float>();
  auto a1=a1b.Get<half>(); auto b1=b1b.Get<half>();
  auto a2=a2b.Get<half>(); auto b2=b2b.Get<half>(); auto c1=c1b.Get<float>();
  GlobalTensor<half> qgm, kgm;
  GlobalTensor<float> dst;
  qgm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query));
  kgm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
  DataCopy(q, qgm, 16 * dim);
  DataCopy(k, kgm, 16 * dim);
  pipe_barrier(PIPE_ALL);
  DataCopyParams pack{16, 1, static_cast<uint16_t>(dim / 16 - 1), 0};
  for (int64_t tile = 0; tile < dim / 16; ++tile) {
    DataCopy(a1[tile * 256], q[tile * 16], pack);
    DataCopy(b1[tile * 256], k[tile * 16], pack);
  }
  pipe_barrier(PIPE_ALL);
  LoadData2DParams load;
  load.startIndex=0; load.repeatTimes=dim / 16;
  load.srcStride=1; load.dstGap=0; load.ifTranspose=false;
  LoadData(a2, a1, load);
  load.ifTranspose=transpose_b;
  LoadData(b2, b1, load);
  pipe_barrier(PIPE_ALL);
  MmadParams mm;
  mm.m=16; mm.n=16; mm.k=dim; mm.cmatrixInitVal=true;
  Mmad(c1, a2, b2, mm);
  pipe_barrier(PIPE_ALL);
  Duplicate(c, 0.0f, 256);
  pipe_barrier(PIPE_ALL);
  DataCopyEnhancedParams enhanced;
  enhanced.blockMode=BlockMode::BLOCK_MODE_MATRIX;
  DataCopy(c, c1, DataCopyParams{1, 1, 0, 0}, enhanced);
  pipe_barrier(PIPE_ALL);
  DataCopy(dst, c, 256);
  pipe_barrier(PIPE_ALL);
}
