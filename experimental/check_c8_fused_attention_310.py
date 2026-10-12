# SPDX-License-Identifier: Apache-2.0
"""Fused C8 attention: quantized/FP16 accuracy, latency samples, mutable Graph."""
import argparse
import json
import statistics
import time
import torch
import torch_npu  # noqa: F401
from c8_cache_310 import C8Cache310

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--cube', action='store_true')
p.add_argument('--splits', type=int, default=1)
p.add_argument('--gqa-groups', type=int, choices=[1, 8, 16], default=8)
p.add_argument('--long', action='store_true')
a = p.parse_args()
torch.ops.load_library(a.library)
torch.manual_seed(97)


def reference(query, keys, values, length):
    queries, heads, dim = query.shape
    kv = keys.shape[1]
    result = []
    for i in range(queries):
        visible = length - queries + i + 1
        q = query[i].reshape(kv, heads // kv, dim).float()
        k = keys[:visible].transpose(0, 1).float()
        v = values[:visible].transpose(0, 1).float()
        scores = torch.matmul(q, k.transpose(1, 2)) * dim ** -.5
        result.append(torch.matmul(scores.softmax(-1), v).reshape(heads, dim))
    return torch.stack(result)


cases = [(1, 1, 128, 64), (3, 3, 256, 64), (65, 2, 256, 64),
         (129, 3, 256, 128), (512, 1, 256, 128), (1024, 1, 128, 128)]
if a.long:
    cases += [(4096, 4, 256, 128), (32768, 1, 256, 128)]
heads = 2 * a.gqa_groups
for length, queries, dim, block in cases:
    pages = (length + block - 1) // block
    cache = C8Cache310(pages + 1, block, 2, dim)
    table_cpu = torch.randperm(pages + 1)[:pages].int()
    table = table_cpu.npu()
    positions = torch.arange(length)
    slots = table_cpu[positions // block].long() * block + positions % block
    keys = torch.randn(length, 2, dim).half()
    values = torch.randn_like(keys)
    if length == 1:
        keys *= 1e-6
        values *= 1e-6
    # Preserve scatter ABI and handle long sequences in bounded writes.
    for start in range(0, length, 4096):
        cache.write(keys[start:start+4096].npu(), values[start:start+4096].npu(),
                    slots[start:start+4096].int().npu())
    query = torch.randn(queries, heads, dim).half().npu()
    actual = cache.decode_causal_fused(query, table, length, cube=a.cube, splits=a.splits).cpu()
    quant_k = cache.k.cpu()[slots].float() * cache.ks.cpu()[slots, :2, None].float()
    quant_v = cache.v.cpu()[slots].float() * cache.vs.cpu()[slots, :2, None].float()
    ref = reference(query.cpu(), quant_k, quant_v, length)
    original = reference(query.cpu(), keys, values, length)
    kernel_error = (actual - ref).abs().max().item()
    quant_error = (actual - original).abs().max().item()
    relative_l2 = ((actual-ref).norm()/ref.norm().clamp_min(1e-12)).item()
    quant_relative_l2 = ((actual-original).norm()/original.norm().clamp_min(1e-12)).item()
    if not torch.isfinite(actual).all() or kernel_error >= .002 or quant_error >= .04:
        print(json.dumps({'failure_tokens': length, 'cube': a.cube,
                          'actual_first_dims': actual[:, :2, :8].tolist(),
                          'reference_first_dims': ref[:, :2, :8].tolist(),
                          'per_query_max_abs': (actual-ref).abs().flatten(1).amax(-1).tolist()}), flush=True)
    assert torch.isfinite(actual).all() and kernel_error < .002 and quant_error < .04, (
        length, queries, kernel_error, quant_error)
    for _ in range(5):
        cache.decode_causal_fused(query, table, length, cube=a.cube, splits=a.splits)
    torch.npu.synchronize()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        cache.decode_causal_fused(query, table, length, cube=a.cube, splits=a.splits)
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    print(json.dumps({'status': 'PASS', 'tokens': length, 'queries': queries,
                      'query_heads': heads, 'kv_heads': 2, 'dim': dim, 'block_size': block,
                      'cube': a.cube, 'splits': a.splits,
                      'kernel_max_abs': kernel_error, 'fp16_max_abs': quant_error,
                      'relative_l2_quantized_reference': relative_l2,
                      'relative_l2_original_fp16': quant_relative_l2,
                      'warmup': 5, 'repeats': 20, 'median_us': statistics.median(times),
                      'samples_us': times, 'timing': 'allocation+launch+synchronize'}), flush=True)

# Capture the final case, then change both query data and physical page order.
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    graph_out = cache.decode_causal_fused(query, table, length, cube=a.cube, splits=a.splits)
for iteration in range(3):
    table_cpu = table_cpu.roll(1)
    table.copy_(table_cpu.npu())
    query.add_(.01)
    graph.replay()
    torch.npu.synchronize()
    slots = table_cpu[positions // block].long() * block + positions % block
    quant_k = cache.k.cpu()[slots].float() * cache.ks.cpu()[slots, :2, None].float()
    quant_v = cache.v.cpu()[slots].float() * cache.vs.cpu()[slots, :2, None].float()
    ref = reference(query.cpu(), quant_k, quant_v, length)
    error = (graph_out.cpu() - ref).abs().max().item()
    assert error < .002, error
    print(json.dumps({'status': 'PASS', 'graph_replay': True,
                      'iteration': iteration, 'mutable_query_and_table': True,
                      'kernel_max_abs': error}), flush=True)
