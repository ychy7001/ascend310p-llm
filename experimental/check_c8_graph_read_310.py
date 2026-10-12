# SPDX-License-Identifier: Apache-2.0
"""Fixed-extent C8 read replay versus eager with mutable tables and lengths."""
import argparse
import json
import torch
import torch_npu
from c8_cache_310 import C8Cache310

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
a = p.parse_args()
torch.ops.load_library(a.library)
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
torch.manual_seed(310)
cache = C8Cache310(4, 64, 2, 256)
key = torch.randn(256, 2, 256, device='npu', dtype=torch.float16)
value = torch.randn_like(key)
cache.write(key, value, torch.arange(256, device='npu', dtype=torch.int32))
table = torch.arange(4, device='npu', dtype=torch.int32)
identity = torch.arange(4, device='npu', dtype=torch.int32).reshape(1, -1)
length = torch.tensor([63], device='npu', dtype=torch.int32)
query = torch.randn(1, 16, 256, device='npu', dtype=torch.float16)
shape = (4, 32, 64, 16)
k = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device='npu', acl_format=29)
v = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device='npu', acl_format=29)
out = torch.empty_like(query)


def forward():
    torch.ops.npu.c8_cache_to_nz(cache.k, cache.v, cache.ks, cache.vs, k, v, 64, table)
    torch_npu._npu_paged_attention(query=query, key_cache=k, value_cache=v,
        num_kv_heads=2, num_heads=16, scale_value=1/16, block_table=identity,
        context_lens=length, out=out)


forward()
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    forward()
for n, order in [(63, [0,1,2,3]), (65, [3,0,2,1]),
                 (127, [1,3,0,2]), (193, [2,1,3,0])]:
    length.fill_(n)
    table.copy_(torch.tensor(order, device='npu', dtype=torch.int32))
    query.add_(0.01)
    forward()
    expected = out.clone()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=.002, rtol=.002)
    print(json.dumps({'status':'PASS', 'length':n, 'table':order,
        'max_abs':(out-expected).abs().max().item(),
        'scope':'C8 conversion plus native PA replay, not model or MTP'}), flush=True)
