# SPDX-License-Identifier: Apache-2.0
"""Selected physical C8 pages -> compact raw NZ, including mutable graph table."""
import argparse
import json
import statistics
import time
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--dim', type=int, choices=[128,256], default=256)
p.add_argument('--block', type=int, choices=[64,128], default=64)
a = p.parse_args()
torch.ops.load_library(a.library)
torch.manual_seed(123)
block, heads, dim, physical = a.block, 2, a.dim, 16
k = torch.randint(-127, 128, (physical*block, heads, dim), dtype=torch.int8).npu()
v = torch.randint(-127, 128, k.shape, dtype=torch.int8).npu()
ks = (torch.rand(physical*block, 16)*.02).half().npu()
vs = (torch.rand(physical*block, 16)*.02).half().npu()
table = torch.tensor([9, 2, 13, 0], dtype=torch.int32).npu()
shape = (4, heads*dim//16, block, 16)
ko = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device='npu', acl_format=29)
vo = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device='npu', acl_format=29)

def convert():
    torch.ops.npu.c8_cache_to_nz(k, v, ks, vs, ko, vo, block, table)

def verify():
    pages = table.cpu().tolist()
    for x, scale, out in ((k, ks, ko), (v, vs, vo)):
        source, scales = x.cpu(), scale.cpu()
        ref = torch.zeros(4, block, heads*dim, dtype=torch.float16)
        for i, page in enumerate(pages):
            if 0 <= page < physical:
                q = source[page*block:(page+1)*block].half()
                s = scales[page*block:(page+1)*block, :heads, None]
                ref[i] = (q*s).reshape(block, heads*dim)
        ref = ref.reshape(4, block, heads*dim//16, 16).permute(0, 2, 1, 3).contiguous()
        error = (out.cpu()-ref).abs().max().item()
        if error != 0:
            print(json.dumps({'nz_actual':out.cpu().flatten()[:32].tolist(),
                              'nz_reference':ref.flatten()[:32].tolist()}),flush=True)
        assert error == 0, error
    print(json.dumps({'status':'PASS', 'physical_pages':physical, 'selected_pages':pages,
                      'max_abs':0, 'logical_nz_bytes':2*ko.numel()*2}), flush=True)

convert()
verify()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    convert()
for pages in ([3, 5, 5, 15], [-1, 16, 7, 4]):
    table.copy_(torch.tensor(pages, dtype=torch.int32).npu())
    ks.mul_(1.01)
    graph.replay()
    torch.npu.synchronize()
    verify()

# PA consumes the compact logical NZ buffer with a compact identity table.
# Invalid physical pages above are only a conversion memory-protection test.
query = torch.randn(1, 16, dim).half().npu()
context = torch.tensor([4*block-3], dtype=torch.int32).npu()
logical = torch.arange(4, dtype=torch.int32).reshape(1, -1).npu()
out = torch.empty_like(query)
table.copy_(torch.tensor([9, 2, 13, 0], dtype=torch.int32).npu())
def attention():
    convert()
    torch_npu._npu_paged_attention(query=query, key_cache=ko, value_cache=vo,
        num_kv_heads=heads, num_heads=16, scale_value=dim**-.5,
        block_table=logical, context_lens=context, out=out)

attention()
pa_graph = torch.npu.NPUGraph()
with torch.npu.graph(pa_graph):
    attention()
for pages in ([9, 2, 13, 0], [3, 5, 5, 15]):
    table.copy_(torch.tensor(pages, dtype=torch.int32).npu())
    query.add_(.01)
    pa_graph.replay()
    torch.npu.synchronize()
    positions = (torch.tensor(pages)[:, None]*block + torch.arange(block)[None, :]).flatten()[:4*block-3]
    qk = k.cpu()[positions].float()*ks.cpu()[positions, :heads, None].float()
    qv = v.cpu()[positions].float()*vs.cpu()[positions, :heads, None].float()
    q = query.cpu().reshape(heads, 8, dim).float()
    scores = q @ qk.transpose(0,1).transpose(1,2)*dim**-.5
    ref = (scores.softmax(-1) @ qv.transpose(0,1)).reshape_as(out.cpu())
    error = (out.cpu().float()-ref).abs().max().item()
    assert error < .002, error
    print(json.dumps({'status':'PASS','selected_pa_graph':True,'pages':pages,
                      'context':4*block-3,'max_abs':error}),flush=True)

for name, fn in [('selected_conversion', convert), ('selected_conversion_pa', attention)]:
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter()-start)*1e6)
    print(json.dumps({'path':name,'warmup':5,'repeats':20,'physical_pages':physical,
                      'selected_pages':4,'median_us':statistics.median(samples),
                      'samples_us':samples,'timing':'launch+synchronize; preallocated buffers'}),flush=True)
