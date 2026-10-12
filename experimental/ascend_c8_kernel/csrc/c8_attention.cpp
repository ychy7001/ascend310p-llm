// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Vector tile implementation. Each core handles one query head; DMA and
// conversion work is batched in 16-token tiles instead of one token at a time.
// K/V remain INT8 in GM; dequant rows and online softmax state stay in UB.
extern "C" __global__ __aicore__ void c8_attention_kernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR table, GM_ADDR output, int64_t queries, int64_t query_heads,
    int64_t kv_heads, int64_t dim, int64_t block_size, int64_t num_blocks,
    int64_t sequence_length) {
  int64_t id = GetBlockIdx();
  if (id >= queries * query_heads) return;
  int64_t query_index = id / query_heads;
  int64_t head = (id % query_heads) / (query_heads / kv_heads);
  int64_t visible = sequence_length - queries + query_index + 1;
  TPipe pipe;
  TBuf<TPosition::VECCALC> qb, qfb, ib, hb, fb, prod, nb, scoreb, redb, tmpb, sb;
  pipe.InitBuffer(qb, 256 * sizeof(half));
  pipe.InitBuffer(qfb, 16 * 256 * sizeof(float));
  pipe.InitBuffer(ib, 16 * 256 * sizeof(int8_t));
  pipe.InitBuffer(hb, 16 * 256 * sizeof(half));
  pipe.InitBuffer(fb, 16 * 256 * sizeof(float));
  pipe.InitBuffer(prod, 16 * 256 * sizeof(float));
  pipe.InitBuffer(nb, 256 * sizeof(float));
  pipe.InitBuffer(scoreb, 128 * sizeof(float));
  pipe.InitBuffer(redb, 64 * sizeof(float));
  pipe.InitBuffer(tmpb, 256 * sizeof(float));
  pipe.InitBuffer(sb, 16 * 16 * sizeof(half));
  auto qh = qb.Get<half>(); auto q = qfb.Get<float>();
  auto xi = ib.Get<int8_t>(); auto xh = hb.Get<half>(); auto x = fb.Get<float>();
  auto product = prod.Get<float>(); auto numerator = nb.Get<float>();
  auto scores = scoreb.Get<float>(); auto reduced = redb.Get<float>();
  auto scratch = tmpb.Get<float>(); auto scales = sb.Get<half>();
  GlobalTensor<half> qgm, ksgm, vsgm;
  GlobalTensor<int8_t> kgm, vgm;
  GlobalTensor<int32_t> mapping;
  GlobalTensor<float> dst;
  qgm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query));
  kgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(key));
  vgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(value));
  ksgm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ks));
  vsgm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(vs));
  mapping.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
  DataCopy(qh, qgm[id * dim], dim);
  pipe_barrier(PIPE_ALL);
  Cast(q, qh, RoundMode::CAST_NONE, dim);
  pipe_barrier(PIPE_V);
  for (int64_t t = 1; t < 16; ++t) Adds(q[t * dim], q, 0.0f, dim);
  Duplicate(numerator, 0.0f, dim);
  pipe_barrier(PIPE_ALL);
  float maximum = -1.0e30f, denominator = 0.0f;
  float factor = dim == 256 ? 0.0625f : 0.08838834764831844f;
  for (int64_t logical = 0; logical * block_size < visible; ++logical) {
    int64_t physical = mapping.GetValue(logical);
    if (physical < 0 || physical >= num_blocks) continue;
    int64_t count = visible - logical * block_size;
    if (count > block_size) count = block_size;
    float next_max = maximum;
    for (int64_t base = 0; base < count; base += 16) {
      int64_t tile = count - base;
      if (tile > 16) tile = 16;
      int64_t slot = physical * block_size + base;
      DataCopyParams dma{static_cast<uint16_t>(tile), static_cast<uint16_t>(dim / 32),
                         static_cast<uint16_t>((kv_heads - 1) * dim / 32), 0};
      DataCopy(xi, kgm[(slot * kv_heads + head) * dim], dma);
      DataCopy(scales, ksgm[slot * 16], tile * 16);
      pipe_barrier(PIPE_ALL);
      Cast(xh, xi, RoundMode::CAST_NONE, tile * dim);
      pipe_barrier(PIPE_V);
      Cast(x, xh, RoundMode::CAST_NONE, tile * dim);
      pipe_barrier(PIPE_V);
      Mul(product, q, x, tile * dim);
      pipe_barrier(PIPE_V);
      WholeReduceSum(reduced, product, 64, tile * dim / 64, 1, 1, 8);
      pipe_barrier(PIPE_ALL);
      for (int64_t t = 0; t < tile; ++t) {
        float sum = 0.0f;
        for (int64_t part = 0; part < dim / 64; ++part)
          sum += reduced.GetValue(t * (dim / 64) + part);
        float scale = static_cast<float>(scales.GetValue(t * 16 + head));
        float score = sum * scale * factor;
        scores.SetValue(base + t, score);
        if (score > next_max) next_max = score;
      }
    }
    // Rescale the previous online state without scalar exp support.
    Duplicate(reduced, maximum - next_max, 8);
    pipe_barrier(PIPE_V);
    Exp(reduced, reduced, 8);
    pipe_barrier(PIPE_ALL);
    float alpha = denominator == 0.0f ? 0.0f : reduced.GetValue(0);
    Muls(numerator, numerator, alpha, dim);
    int64_t padded = (count + 7) / 8 * 8;
    for (int64_t t = count; t < padded; ++t) scores.SetValue(t, -1.0e30f);
    pipe_barrier(PIPE_ALL);
    Adds(scores, scores, -next_max, padded);
    pipe_barrier(PIPE_V);
    Exp(scores, scores, padded);
    pipe_barrier(PIPE_V);
    ReduceSum(reduced, scores, scratch, padded);
    pipe_barrier(PIPE_ALL);
    denominator = denominator * alpha + reduced.GetValue(0);
    for (int64_t base = 0; base < count; base += 16) {
      int64_t tile = count - base;
      if (tile > 16) tile = 16;
      int64_t slot = physical * block_size + base;
      DataCopyParams dma{static_cast<uint16_t>(tile), static_cast<uint16_t>(dim / 32),
                         static_cast<uint16_t>((kv_heads - 1) * dim / 32), 0};
      DataCopy(xi, vgm[(slot * kv_heads + head) * dim], dma);
      DataCopy(scales, vsgm[slot * 16], tile * 16);
      pipe_barrier(PIPE_ALL);
      Cast(xh, xi, RoundMode::CAST_NONE, tile * dim);
      pipe_barrier(PIPE_V);
      Cast(x, xh, RoundMode::CAST_NONE, tile * dim);
      pipe_barrier(PIPE_V);
      for (int64_t t = 0; t < tile; ++t) {
        float probability = scores.GetValue(base + t);
        float scale = static_cast<float>(scales.GetValue(t * 16 + head));
        Muls(x[t * dim], x[t * dim], scale * probability, dim);
        pipe_barrier(PIPE_V);
        Add(numerator, numerator, x[t * dim], dim);
        pipe_barrier(PIPE_V);
      }
      pipe_barrier(PIPE_ALL);
    }
    maximum = next_max;
  }
  if (denominator > 0.0f) Muls(numerator, numerator, 1.0f / denominator, dim);
  pipe_barrier(PIPE_ALL);
  DataCopy(dst[id * dim], numerator, dim);
  pipe_barrier(PIPE_ALL);
}
