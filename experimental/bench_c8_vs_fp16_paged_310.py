# SPDX-License-Identifier: Apache-2.0
"""Compare fused C8 vector prototype with the native 310P FP16 paged op."""
import argparse
import json
import statistics
import time
import numpy as np
import torch
import torch_npu
from c8_cache_310 import C8Cache310

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--cube', action='store_true')
p.add_argument('--splits', type=int, default=1)
p.add_argument('--staged', action='store_true', help='Dequantize C8 into reusable NZ buffers, then native PA')
p.add_argument('--profile-dir', help='Optional final 4096-token native/C8 NPU profile')
p.add_argument('--lengths', type=int, nargs='+', default=[128, 512, 1024, 4096])
a = p.parse_args()
if any(n <= 0 or n % 64 for n in a.lengths):
    p.error('--lengths must contain positive multiples of 64')
torch.ops.load_library(a.library)
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = True
torch.manual_seed(101)
for length in a.lengths:
    # 310P native PA requires block_size * head_dim <= 128 * 128.
    block, kv, heads, dim = 64, 2, 16, 256
    blocks = length // block
    # Generate on CPU: CANN 9.1/310P may route torch.randn(device=NPU)
    # through StatelessRandomNormalV2, which is unavailable in some
    # standalone containers even though the tested attention kernels work.
    keys = torch.from_numpy(np.random.default_rng(101 + length).standard_normal(
        (length, kv, dim)).astype(np.float16)).npu()
    values = torch.from_numpy(np.random.default_rng(202 + length).standard_normal(
        (length, kv, dim)).astype(np.float16)).npu()
    query = torch.from_numpy(np.random.default_rng(303 + length).standard_normal(
        (1, heads, dim)).astype(np.float16)).npu()
    table = torch.arange(blocks, dtype=torch.int32).npu()
    slots = torch.arange(length, dtype=torch.int32).npu()
    context = torch.tensor([length], dtype=torch.int32).npu()
    c8 = C8Cache310(blocks, block, kv, dim)
    for start in range(0, length, 4096):
        c8.write(keys[start:start+4096], values[start:start+4096], slots[start:start+4096])
    # Official 310P cache contract is raw NZ [blocks, hidden/16, block, 16].
    k16 = torch_npu.empty_with_format(size=(blocks, kv * dim // 16, block, 16),
                                     dtype=torch.float16, device='npu', acl_format=29)
    v16 = torch_npu.empty_with_format(size=k16.shape, dtype=torch.float16,
                                     device='npu', acl_format=29)
    torch_npu._npu_reshape_and_cache(key=keys,
                                   value=values,
                                   key_cache=k16, value_cache=v16, slot_indices=slots)
    native_out = torch.empty_like(query)
    if a.staged:
        staged_k = torch_npu.empty_with_format(size=k16.shape, dtype=torch.float16, device='npu', acl_format=29)
        staged_v = torch_npu.empty_with_format(size=k16.shape, dtype=torch.float16, device='npu', acl_format=29)
        staged_out = torch.empty_like(query)

        def staged():
            torch.ops.npu.c8_cache_to_nz(c8.k, c8.v, c8.ks, c8.vs, staged_k, staged_v, block)
            torch_npu._npu_paged_attention(query=query, key_cache=staged_k, value_cache=staged_v,
                num_kv_heads=kv, num_heads=heads, scale_value=dim ** -.5,
                block_table=table.reshape(1, -1), context_lens=context, out=staged_out)
            return staged_out.reshape_as(query)

    def fp16():
        torch_npu._npu_paged_attention(query=query,
            key_cache=k16, value_cache=v16, num_kv_heads=kv, num_heads=heads,
            scale_value=dim ** -.5, block_table=table.reshape(1, -1),
            context_lens=context, out=native_out)
        return native_out.reshape_as(query)

    cpu_q = query.cpu().reshape(kv, heads // kv, dim).float()
    cpu_k = keys.cpu().transpose(0, 1).float()
    cpu_v = values.cpu().transpose(0, 1).float()
    reference = torch.matmul((torch.matmul(cpu_q, cpu_k.transpose(1, 2)) * dim ** -.5).softmax(-1),
                             cpu_v).reshape_as(query.cpu())
    paths = [('native_fp16_paged', fp16),
                     ('fused_c8_cube' if a.cube else 'fused_c8_vector',
                      lambda: c8.decode_causal_fused(query, table, length, cube=a.cube, splits=a.splits))]
    if a.staged:
        paths.append(('staged_c8_native_pa', staged))
    for name, fn in paths:
        actual = fn().cpu().float()
        error = (actual - reference).abs().max().item()
        assert error < .04 and torch.isfinite(actual).all(), (length, name, error)
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            fn()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1e6)
        print(json.dumps({'status': 'PASS', 'path': name, 'tokens': length,
                          'splits': a.splits,
                          'query_heads': heads, 'kv_heads': kv, 'dim': dim,
                          'warmup': 5, 'repeats': 20, 'max_abs_vs_fp16_reference': error,
                          'median_us': statistics.median(samples), 'samples_us': samples,
                          'timing': ('host launch+synchronize; output/scratch allocation' if name.startswith('fused_c8')
                                     else 'host launch+synchronize; preallocated output and NZ buffers'),
                          'temporary_nz_bytes': (staged_k.numel()+staged_v.numel())*2 if name=='staged_c8_native_pa' else 0,
                          'kv_storage_bytes': c8.storage_bytes if name.startswith('fused_c8') else
                          (c8.storage_bytes if name=='staged_c8_native_pa' else (k16.numel() + v16.numel()) * 2)}), flush=True)

if a.staged:
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = staged()
    query.add_(.01)
    table.copy_(table.roll(1))
    c8.ks.mul_(1.01)
    graph.replay()
    torch.npu.synchronize()
    order = table.cpu().long()[:, None]*block + torch.arange(block)[None, :]
    slots_cpu = order.flatten()
    qk = c8.k.cpu()[slots_cpu].float()*c8.ks.cpu()[slots_cpu, :kv, None].float()
    qv = c8.v.cpu()[slots_cpu].float()*c8.vs.cpu()[slots_cpu, :kv, None].float()
    q = query.cpu().reshape(kv,heads//kv,dim).float()
    scores = torch.matmul(q,qk.transpose(0,1).transpose(1,2))*dim**-.5
    ref = torch.matmul(scores.softmax(-1),qv.transpose(0,1)).reshape_as(captured.cpu())
    error = (captured.cpu().float()-ref).abs().max().item()
    assert error < .002, error
    print(json.dumps({'status':'PASS','path':'staged_c8_graph','tokens':length,
                      'mutable_query_table_and_scale':True,'max_abs_vs_quant_reference':error}),flush=True)

if a.profile_dir:
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(a.profile_dir),
    ) as profiler:
        for _ in range(5):
            with torch.profiler.record_function(f'native_fp16_paged_{length}'):
                fp16()
            with torch.profiler.record_function(f'fused_c8_{length}'):
                c8.decode_causal_fused(query, table, length, cube=a.cube, splits=a.splits)
            if a.staged:
                with torch.profiler.record_function(f'staged_c8_{length}'):
                    staged()
        torch.npu.synchronize()
        profiler.step()
    print(json.dumps({'profile_dir': a.profile_dir, 'profile_tokens': length,
                      'profile_repeats': 5}), flush=True)
