# SPDX-License-Identifier: Apache-2.0
"""Validate and time the direct single-page C8 -> persistent NZ prototype."""
import argparse
import json
import statistics
import time

import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--max-length', type=int, default=114688)
p.add_argument('--repeats', type=int, default=20)
a = p.parse_args()
if a.max_length % 64 or a.repeats < 1:
    p.error('max length must be a positive multiple of 64')
torch.ops.load_library(a.library)
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
slots = a.max_length
k = torch.randint(-127, 128, (slots, 2, 256), device='npu', dtype=torch.int8)
v = torch.randint_like(k, -127, 128)
ks = torch.ones(slots, 16, device='npu', dtype=torch.float16)
vs = torch.ones_like(ks)
full_table = torch.arange(slots // 64, device='npu', dtype=torch.int32)
one_table = torch.tensor([slots // 64 - 1], device='npu', dtype=torch.int32)
full_shape = (slots // 64, 32, 64, 16)
full_k = torch_npu.empty_with_format(size=full_shape, dtype=torch.float16, device='npu', acl_format=29)
full_v = torch_npu.empty_with_format(size=full_shape, dtype=torch.float16, device='npu', acl_format=29)
length = torch.tensor([slots], device='npu', dtype=torch.int32)

def full():
    torch.ops.npu.c8_cache_to_nz(k, v, ks, vs, full_k, full_v, 64, full_table, length)

def direct_page():
    torch.ops.npu.c8_cache_to_nz_range(k, v, ks, vs, full_k, full_v, 64, one_table,
                                       slots // 64 - 1)

for _ in range(5):
    full()
torch.npu.synchronize()
full_k_ref = full_k.clone()
full_v_ref = full_v.clone()
torch.npu.synchronize()

# Preserve a sentinel in every page, then direct-write only the final page.
sentinel = torch.full_like(full_k, 7)
full_k.copy_(sentinel)
full_v.copy_(sentinel)
direct_page()
torch.npu.synchronize()
last_k_err = (full_k[-1:] - full_k_ref[-1:]).abs().max().item()
last_v_err = (full_v[-1:] - full_v_ref[-1:]).abs().max().item()
prefix_k_max = full_k[:-1].abs().max().item()
prefix_v_max = full_v[:-1].abs().max().item()

for _ in range(5):
    full()
    direct_page()
torch.npu.synchronize()
samples = {'full_114k': [], 'direct_one_page': []}
for i in range(a.repeats):
    order = (('full_114k', full), ('direct_one_page', direct_page))
    if i % 2:
        order = tuple(reversed(order))
    for name, fn in order:
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples[name].append((time.perf_counter() - start) * 1e6)
print(json.dumps({
    'max_length': slots,
    'median_us': {name: statistics.median(values) for name, values in samples.items()},
    'samples_us': samples,
    'last_page_max_abs_error': {'k': last_k_err, 'v': last_v_err},
    'untouched_prefix_abs_max': {'k': prefix_k_max, 'v': prefix_v_max},
    'protocol': 'direct page write into preallocated full NZ workspace',
}), flush=True)
