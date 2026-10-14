# SPDX-License-Identifier: Apache-2.0
"""Estimate a one-page C8 NZ update using the existing selected-page ABI.

This is a kernel-only upper-bound probe. It does not implement the copy-back
needed to place a single converted page into a persistent full NZ workspace.
"""
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
one_shape = (1, 32, 64, 16)
full_k = torch_npu.empty_with_format(size=full_shape, dtype=torch.float16, device='npu', acl_format=29)
full_v = torch_npu.empty_with_format(size=full_shape, dtype=torch.float16, device='npu', acl_format=29)
one_k = torch_npu.empty_with_format(size=one_shape, dtype=torch.float16, device='npu', acl_format=29)
one_v = torch_npu.empty_with_format(size=one_shape, dtype=torch.float16, device='npu', acl_format=29)
length = torch.tensor([slots], device='npu', dtype=torch.int32)

def full():
    torch.ops.npu.c8_cache_to_nz(k, v, ks, vs, full_k, full_v, 64, full_table, length)

def one():
    torch.ops.npu.c8_cache_to_nz(k, v, ks, vs, one_k, one_v, 64, one_table, torch.tensor([64], device='npu', dtype=torch.int32))

def one_with_copy():
    one()
    full_k[-1:].copy_(one_k)
    full_v[-1:].copy_(one_v)

for fn in (full, one):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
samples = {'full_114k': [], 'one_page': [], 'one_page_copyback': []}
for i in range(a.repeats):
    order = (('full_114k', full), ('one_page', one), ('one_page_copyback', one_with_copy))
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
    'protocol': 'preallocated output; selected-page ABI; synchronized host replay',
    'limitation': 'one_page output is not placed into a persistent full NZ workspace',
}), flush=True)
