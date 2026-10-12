# SPDX-License-Identifier: Apache-2.0
"""Fixed-extent graph conversion microbenchmark; not model throughput or pure kernel time."""
import argparse
import json
import math
import statistics
import time

import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', required=True)
p.add_argument('--max-length', type=int, default=131072)
p.add_argument('--lengths', type=int, nargs='+', default=[128, 8192, 32768, 131072])
p.add_argument('--repeats', type=int, default=30)
a = p.parse_args()
if a.max_length % 64 or min(a.lengths) < 1 or max(a.lengths) > a.max_length or a.repeats < 1:
    p.error('require positive lengths within a page-aligned max and positive repeats')
torch.ops.load_library(a.library)
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
slots = a.max_length
k = torch.randint(-127, 128, (slots, 2, 256), device='npu', dtype=torch.int8)
v = torch.randint_like(k, -127, 128)
ks = torch.ones(slots, 16, device='npu', dtype=torch.float16)
vs = torch.ones_like(ks)
table = torch.arange(slots // 64, device='npu', dtype=torch.int32)
length = torch.tensor([128], device='npu', dtype=torch.int32)
shape = (slots // 64, 32, 64, 16)
ko = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device='npu', acl_format=29)
vo = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device='npu', acl_format=29)


def forward(limited):
    torch.ops.npu.c8_cache_to_nz(k, v, ks, vs, ko, vo, 64, table, length if limited else None)


graphs = {}
for limited in (False, True):
    for _ in range(5):
        forward(limited)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward(limited)
    graphs[limited] = graph
for n in a.lengths:
    length.fill_(n)
    forward(False)
    pages = math.ceil(n / 64)
    reference_k, reference_v = ko[:pages].clone(), vo[:pages].clone()
    ko.fill_(float('nan'))
    vo.fill_(float('nan'))
    graphs[True].replay()
    torch.npu.synchronize()
    torch.testing.assert_close(ko[:pages].cpu(), reference_k.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(vo[:pages].cpu(), reference_v.cpu(), atol=0, rtol=0)
    times = {False: [], True: []}
    for graph in graphs.values():
        for _ in range(5):
            graph.replay()
    torch.npu.synchronize()
    # Alternate order to reduce drift; keep every synchronized replay sample.
    for repeat in range(a.repeats):
        for limited in ((False, True) if repeat % 2 == 0 else (True, False)):
            start = time.perf_counter()
            graphs[limited].replay()
            torch.npu.synchronize()
            times[limited].append((time.perf_counter() - start) * 1e6)
    print(json.dumps({'length': n, 'max_length': slots, 'valid_pages_exact': True,
        'full_us': times[False], 'device_length_us': times[True],
        'median_full_us': statistics.median(times[False]),
        'median_device_length_us': statistics.median(times[True]),
        'protocol': '5 warmup; alternating synchronized graph replay host time; conversion only'}), flush=True)
