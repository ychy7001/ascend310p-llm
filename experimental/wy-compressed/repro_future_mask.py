#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""310P causal invariant: changing future-only V must not change a query row."""
import argparse
import json

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    device = f'npu:{args.device}'
    block, heads, dim, q_len, row = 64, 8, 256, 384, 123
    q = torch.zeros((q_len, heads, dim), dtype=torch.float16, device=device)
    mask = torch.full((2048, 2048), -10000., dtype=q.dtype, device=device).triu_(1)
    failures = []
    for context in (32768, 32774):
        pages = (context + block - 1) // block
        capacity = pages * block
        cutoff = context - q_len + row
        cache_shape = (pages, dim // 16, block, 16)
        cache = [torch_npu.empty_with_format(size=cache_shape, dtype=q.dtype, device=device, acl_format=29)
                 for _ in range(2)]
        k = torch.zeros((capacity, 1, dim), dtype=q.dtype, device=device)
        v = torch.zeros_like(k)
        slots = torch.arange(capacity, dtype=torch.int32, device=device)
        table = torch.arange(pages, dtype=torch.int32, device=device).reshape(1, -1)
        lens = torch.tensor([context], dtype=torch.int32, device=device)
        width = (context + 15) // 16 * 16
        positions = torch.arange(context - q_len, context, device=device)
        legacy_mask = torch.zeros((q_len, width), dtype=q.dtype, device=device)
        legacy_mask.masked_fill_(torch.arange(width, device=device)[None, :] > positions[:, None], -torch.inf)
        nz = legacy_mask.reshape(1, q_len, width // 16, 16).permute(0, 2, 1, 3).contiguous()
        nz = torch_npu.npu_format_cast(nz, 29)
        results = {'legacy': {}}
        operators = [('legacy', torch_npu._npu_paged_attention_splitfuse, nz, {})]
        if hasattr(torch_npu, '_npu_paged_attention_splitfuse_v2'):
            results['compressed'] = {}
            operators.append(('compressed', torch_npu._npu_paged_attention_splitfuse_v2, mask, {'mask_type': 5}))
        for name, value in [('zero', 0.), ('future', 64.)]:
            v[cutoff + 1:cutoff + 15].fill_(value)
            for i in range(0, capacity, 3840):
                sl = slice(i, i + 3840)
                torch_npu._npu_reshape_and_cache(key=k[sl], value=v[sl], key_cache=cache[0],
                                                value_cache=cache[1], slot_indices=slots[sl])
            for arm, op, attn_mask, extra in operators:
                out = torch.empty_like(q)
                op(query=q, key_cache=cache[0], value_cache=cache[1], mask=attn_mask, block_table=table,
                   seq_len=torch.tensor([q_len], dtype=torch.int32), context_lens=lens,
                   num_kv_heads=1, num_heads=heads, scale_value=dim ** -.5, out=out, **extra)
                samples = out[[row, q_len - 1]].cpu().float()
                results[arm][name] = dict(row=float(samples[0].abs().max()), last=float(samples[1, 0, 0]))
        expected_last = 14 * 64 / context
        for arm in results.values():
            assert arm['zero']['last'] == 0 and abs(arm['future']['last'] - expected_last) < 0.0003, results
        assert results['legacy']['zero']['row'] == results['legacy']['future']['row'] == 0, results
        tested = results.get('compressed', results['legacy'])
        unchanged = tested['zero']['row'] == tested['future']['row'] == 0
        print(json.dumps(dict(device=args.device, torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
              query=q_len, context=context, physical_block=block, row=row, causal_cutoff=cutoff,
              results=results, compressed_available='compressed' in results, future_invariant=unchanged)), flush=True)
        if not unchanged:
            failures.append(context)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
