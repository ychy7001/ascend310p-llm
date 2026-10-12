# SPDX-License-Identifier: Apache-2.0
"""Validate device quantization, padded slots, decode append, and page reads."""
import json
import argparse
import torch
import torch_npu  # noqa: F401
from c8_cache_310 import C8Cache310

p = argparse.ArgumentParser()
p.add_argument('--library', default='/tmp/libqwen36_c8.so')
a = p.parse_args()
torch.ops.load_library(a.library)
torch.manual_seed(31)
cache = C8Cache310(4, 64, 2, 256)
k = torch.randn(130, 2, 256).half()
v = torch.randn_like(k)
table = torch.tensor([2, 0, 3])
pos = torch.arange(129)
slots = table[pos // 64] * 64 + pos % 64
padded = torch.cat((slots[:128], torch.tensor([-1])))
cache.write(k[:129].npu(), v[:129].npu(), padded.npu())
torch.npu.synchronize()
torch.testing.assert_close(cache.k.cpu()[192:], torch.zeros_like(cache.k.cpu()[192:]), atol=0, rtol=0)
# Decode append fills only the first slot in the last page.
cache.write(k[128:129].npu(), v[128:129].npu(), slots[128:].npu())
torch.npu.synchronize()
errors = []
for source, stored, scales in ((k[:129], cache.k, cache.ks), (v[:129], cache.v, cache.vs)):
    s = (source.float().abs().amax(-1) / 127).clamp_min(torch.finfo(torch.float16).tiny).half().float()
    quant = (source.float() / s.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    # Device rounding may differ at exact ties; tolerate one quantization bin.
    assert (stored.cpu()[slots].short() - quant.short()).abs().max().item() <= 1
    torch.testing.assert_close(scales.cpu()[slots, :2], s.half(), atol=2e-5, rtol=.002)
    errors.append((stored.cpu()[slots].float() * scales.cpu()[slots, :2, None] - source.float()).abs().max().item())
pages = [cache.read_page(int(p), min(64, 129-i*64)) for i, p in enumerate(table)]
torch.npu.synchronize()
for which, source in enumerate((k[:129], v[:129])):
    actual = torch.cat([pair[which].cpu() for pair in pages])
    torch.testing.assert_close(actual.float(), source.float(), atol=.04, rtol=.03)
print(json.dumps({'status': 'PASS', 'device_quantization': True, 'padding_slot': -1,
                  'decode_append': True, 'max_abs_kv_error': errors,
                  'cache_bytes': cache.storage_bytes}))

# Real GQA head shapes, shuffled physical pages, tails, and near-zero scales.
for dim in (128, 256):
    for length in (1, 65, 129):
        candidate = C8Cache310(4, 64, 2, dim)
        keys = torch.randn(length, 2, dim).half()
        values = torch.randn_like(keys)
        if length == 1:
            keys *= 1e-6
            values *= 1e-6
        positions = torch.arange(length)
        mapping = table[positions // 64] * 64 + positions % 64
        candidate.write(keys.npu(), values.npu(), mapping.npu())
        query = torch.randn(16, dim).half()
        actual = candidate.decode_reference(query.npu(), table.tolist(), length).cpu()
        q = query.reshape(2, 8, dim).float()
        scores = torch.matmul(q, keys.transpose(0, 1).float().transpose(1, 2)) * dim ** -.5
        ref = torch.matmul(scores.softmax(-1), values.transpose(0, 1).float()).reshape(16, dim)
        error = (actual-ref).abs().max().item()
        assert torch.isfinite(actual).all() and error < .04, (dim, length, error)
        assert candidate.ks.cpu()[mapping, :2].min().item() > 0
        print(json.dumps({'status':'PASS', 'path':'page-local GQA reference',
                          'head_dim':dim, 'query_heads':16, 'kv_heads':2,
                          'tokens':length, 'near_zero':length == 1, 'max_abs':error}))

# MTP verify uses multiple causal queries, including a query span crossing pages.
for length, queries in ((3, 3), (65, 2), (129, 3)):
    candidate = C8Cache310(4, 64, 2, 256)
    keys = torch.randn(length, 2, 256).half()
    values = torch.randn_like(keys)
    positions = torch.arange(length)
    mapping = table[positions // 64] * 64 + positions % 64
    candidate.write(keys.npu(), values.npu(), mapping.npu())
    query = torch.randn(queries, 16, 256).half()
    actual = candidate.decode_causal_reference(query.npu(), table.tolist(), length).cpu()
    reference = []
    for i in range(queries):
        visible = length - queries + i + 1
        q = query[i].reshape(2, 8, 256).float()
        scores = torch.matmul(q, keys[:visible].transpose(0, 1).float().transpose(1, 2)) / 16
        reference.append(torch.matmul(scores.softmax(-1),
                                     values[:visible].transpose(0, 1).float()).reshape(16, 256))
    error = (actual - torch.stack(reference)).abs().max().item()
    assert torch.isfinite(actual).all() and error < .04, (length, queries, error)
    print(json.dumps({'status': 'PASS', 'path': 'C8 causal MTP verify reference',
                      'tokens': length, 'queries': queries, 'max_abs': error}))
