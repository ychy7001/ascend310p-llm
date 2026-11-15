#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure causal prefix length for every query row with zero Q/K and coded V."""
import json
import os
import sys

import torch
import torch_npu


def ranges(values):
    runs = []
    for i, value in enumerate(values):
        if runs and runs[-1][2] == value:
            runs[-1][1] = i
        else:
            runs.append([i, i, value])
    return runs


def main():
    torch.set_num_threads(1)
    rank = int(sys.argv[1])
    torch.npu.set_device(rank)
    device = f'npu:{rank}'
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    dtype = torch.float16
    block = int(sys.argv[2])
    heads = 8
    dim = int(os.getenv('CUTOFF_HEAD_DIM', "128" if block == 128 else "256"))
    mask = torch.full((2048, 2048), -10000., dtype=dtype, device=device).triu_(1)
    cases = {(c, 384) for base in (256, 1024, 2048, 4096, 32768)
             for c in range(max(384, base - 1), base + 3)}
    cases |= {(c, q) for c in (32704, 32767, 32768, 32769, 32774, 32776, 32784, 32799,
                               32800, 32801, 32831, 32832, 32833)
              for q in (1, 3, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 383, 384, 385)}
    cases |= {(c, 384) for c in range(32768, 32833)}
    for c in (1024, 2048, 8192, 32768):
        cases |= {(c + tail, 64) for tail in (0, 1, 6, 16, 32, 63)}
    grid = os.getenv('CUTOFF_GRID')
    if grid == 'boundary':
        cases = {(c, q) for c in range(2044, 2057) for q in (3, 64, 384, 1024)}
        cases |= {(2048 + q, q) for q in (3, 64, 384, 1024, 1920)}
        cases |= {(2049 + q, q) for q in (3, 64, 384, 1024, 1920)}
        cases |= {(c, q) for c in (4095, 4096, 4097, 32768, 32774)
                  for q in (64, 384, 1920)}
    if grid in ('layout', 'geometry'):
        cases = {(2048, 384), (2049, 384), (2054, 384), (32768, 383), (32768, 384),
                 (32774, 384), (32784, 16), (32784, 384), (32774, 1920)}
        if grid == 'layout':
            mask = torch.full_like(mask, -65504.).triu_(1)
    for context, qlen in sorted(cases):
        pages = (context + block - 1) // block
        capacity = pages * block
        shape = (pages, dim // 16, block, 16)
        caches = [torch_npu.empty_with_format(size=shape, dtype=dtype, device=device, acl_format=29)
                  for _ in range(2)]
        k = torch.zeros((capacity, 1, dim), dtype=dtype, device=device)
        slots = torch.arange(capacity, dtype=torch.int32, device=device)
        table = torch.arange(pages, dtype=torch.int32, device=device).reshape(1, -1)
        if grid == 'layout':
            table = table.flip(-1)
            slots = table.flatten().repeat_interleave(block) * block + slots % block
        q = torch.zeros((qlen, heads, dim), dtype=dtype, device=device)
        lens = torch.tensor([context], dtype=torch.int32, device=device)
        qlens = torch.tensor([qlen], dtype=torch.int32)
        cuts = torch.arange(context - qlen, context)
        # Each key in the query suffix gets a unique V channel per pass. A visible
        # marker contributes at least 64/context; threshold at half that value.
        # This measures the visible set, including holes, without fitting a prefix.
        extra = [[] for _ in range(qlen)]
        missing = [[] for _ in range(qlen)]
        legacy_control = (context, qlen) in {(1024, 384), (2049, 64), (2049, 384), (32768, 383), (32774, 384), (32784, 384)}
        if legacy_control:
            width, height = (context + 15) // 16 * 16, (qlen + 15) // 16 * 16
            causal = torch.zeros((height, width), dtype=dtype, device=device)
            causal[:qlen].masked_fill_(torch.arange(width, device=device)[None, :] > cuts.to(device)[:, None], -torch.inf)
            nz = causal.reshape(1, height, width // 16, 16).permute(0, 2, 1, 3).contiguous()
            nz = torch_npu.npu_format_cast(nz, 29)
        for start in range(context - qlen, context, dim):
            stop = min(context, start + dim)
            ids = torch.arange(start, stop)
            v = torch.zeros_like(k)
            v[ids.to(device), 0, torch.arange(stop-start, device=device)] = 64.
            for i in range(0, capacity, 3840):
                sl = slice(i, i + 3840)
                torch_npu._npu_reshape_and_cache(key=k[sl], value=v[sl], key_cache=caches[0],
                                                value_cache=caches[1], slot_indices=slots[sl])
            out = torch.empty_like(q)
            args = dict(query=q, key_cache=caches[0], value_cache=caches[1], block_table=table,
                        seq_len=qlens, context_lens=lens, num_kv_heads=1, num_heads=heads,
                        scale_value=dim ** -.5, out=out)
            expected = ids[None, :] <= cuts[:, None]
            if legacy_control:
                torch_npu._npu_paged_attention_splitfuse(mask=nz, **args)
                observed = out[:, :, :stop-start].cpu().float() > (32. / context)
                assert torch.equal(observed, expected[:, None, :].expand_as(observed)), (context, qlen, start)
            torch_npu._npu_paged_attention_splitfuse_v2(mask=mask, mask_type=5, **args)
            values = out[:, :, :stop-start].cpu().float()
            assert torch.isfinite(values).all()
            observed = values > (32. / context)
            assert torch.equal(observed, observed[:, :1, :].expand_as(observed)), (context, qlen, start)
            visible = observed[:, 0, :]
            for row, col in (visible & ~expected).nonzero().tolist():
                extra[row].append(start + col - int(cuts[row]))
            for row, col in (~visible & expected).nonzero().tolist():
                missing[row].append(start + col - int(cuts[row]))
        affected = [i for i in range(qlen) if extra[i] or missing[i]]
        print(json.dumps(dict(rank=rank, block=block, head_dim=dim, context=context, query=qlen,
            history=context-qlen, legacy_control=legacy_control,
            extra_count_runs=ranges(list(map(len, extra))), missing_count_runs=ranges(list(map(len, missing))),
            affected_rows=len(affected), examples={i:dict(extra=extra[i], missing=missing[i])
                for i in sorted(set(affected[:10] + affected[-3:] + ([123] if qlen > 123 else [])))})), flush=True)



if __name__ == '__main__':
    main()
