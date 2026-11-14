#!/usr/bin/env python3
"""Compare compressed and active-width legacy SplitFuse with the same real inputs."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu

root = Path('/vllm-workspace/vllm-ascend/vllm_ascend')
ACL_FORMAT_FRACTAL_NZ = 29


def load_fn(path, name, cls=None):
    nodes = ast.parse(path.read_text()).body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), globals())
    return globals()[name]


load_fn(root / 'utils.py', 'nd_to_nz_spec')
# Frozen release's legacy builder body, not an approximation of its mask.
legacy = load_fn(Path('/bench/legacy_attention_mask.py'), 'get_splitfuse_mask', 'AttentionMaskBuilder310')
mask_source = root / '_310p/attention/attention_mask.py'
text = ast.parse(mask_source.read_text())
for node in text.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
       ('PAGED_ATTENTION_COMPRESSED_MASK_VALUE', 'COMPRESSED_MASK_SEQ_LEN') for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(mask_source), 'exec'), globals())
compressed = load_fn(mask_source, 'get_compressed_splitfuse_mask', 'AttentionMaskBuilder310')
# Public code uses this mask_type for compressed paged attention.
attn_source = ast.parse((root / '_310p/attention/attention_v1.py').read_text())
for node in attn_source.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
       t.id == 'MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION' for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<mask_type>', 'exec'), globals())


def main():
    import sys
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    torch.npu.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    block, heads, kv, dim = (int(sys.argv[2]) if len(sys.argv) > 2 else 64), 8, 1, 256
    layouts = [([1920], [3840]), ([1920], [32768]),
               ([512, 384, 1024], [12544, 32774, 1024]),
               ([384, 1024, 512], [32774, 1024, 12544]),
               ([3, 3, 3, 3], [128, 1024, 32768, 4096])]
    # Keep the original RNG sequence, then isolate context-tail alignment from batching.
    layouts += [([384], [c]) for c in (32768, 32769, 32774, 32784, 32832, 32767)]
    layouts += [([3], [c]) for c in (32768, 32769, 32774, 32784)]
    failures = []
    capacity = max(sum(((c + block - 1) // block) * block for c in cs) for _, cs in layouts)
    shape = (capacity // block, kv * dim // 16, block, 16)
    kc = torch_npu.empty_with_format(size=shape, device='npu', dtype=torch.float16, acl_format=29)
    vc = torch_npu.empty_with_format(size=shape, device='npu', dtype=torch.float16, acl_format=29)
    k_cpu = torch.randn((capacity, kv, dim)).half()
    v_cpu = torch.randn_like(k_cpu)
    k, v = k_cpu.npu(), v_cpu.npu()
    slots = torch.arange(capacity, dtype=torch.int32).npu()
    for i in range(0, capacity, 3840):
        sl = slice(i, i + 3840)
        torch_npu._npu_reshape_and_cache(key=k[sl], value=v[sl], key_cache=kc, value_cache=vc,
                                        slot_indices=slots[sl])
    holder = SimpleNamespace(compressed_chunked_prefill_attn_mask=None)
    fixed = compressed(holder, k.device)
    for qlens, contexts in layouts:
        starts = [0]
        for ql in qlens:
            starts.append(starts[-1] + ql)
        ctx = torch.tensor(contexts, dtype=torch.int32).npu()
        meta = SimpleNamespace(query_start_loc=torch.tensor(starts, dtype=torch.int32).npu(), seq_lens=ctx)
        mask = legacy(None, meta, ctx.device)
        table = torch.zeros((len(contexts), (max(contexts) + block - 1) // block), dtype=torch.int32)
        page = 0
        offsets = []
        for i, c in enumerate(contexts):
            offsets.append(page * block)
            count = (c + block - 1) // block
            table[i, :count] = torch.arange(page, page + count)
            page += count
        table = table.npu()
        for trial in range(2):
            qc = torch.randn((starts[-1], heads, dim)).half()
            q = qc.npu()
            results = []
            for op, m, extra in [(torch_npu._npu_paged_attention_splitfuse, mask, {}),
                                  (torch_npu._npu_paged_attention_splitfuse_v2, fixed,
                                   {'mask_type': MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION})]:
                out = torch.empty_like(q)
                op(query=q, key_cache=kc, value_cache=vc, mask=m, block_table=table,
                   seq_len=torch.tensor(qlens, dtype=torch.int32), context_lens=ctx,
                   num_kv_heads=kv, num_heads=heads, scale_value=dim ** -.5, out=out, **extra)
                results.append(out.cpu())
            delta = (results[0].float() - results[1].float()).abs()
            errors = [0., 0.]
            worst = []
            for i, (ql, c, offset) in enumerate(zip(qlens, contexts, offsets)):
                row_delta = delta[starts[i]:starts[i+1]].reshape(ql,-1).amax(-1)
                top = torch.topk(row_delta, min(3, ql)).indices.tolist()
                local = torch.tensor(sorted(set((0, ql // 2, ql - 1, *top))))
                idx = starts[i] + local
                scores = (qc[idx].float() @ k_cpu[offset:offset+c, 0].float().T) * dim ** -.5
                scores.masked_fill_(torch.arange(c)[None, None, :] > (c - ql + local)[:, None, None], -float('inf'))
                ref = scores.softmax(-1) @ v_cpu[offset:offset+c, 0].float()
                for arm, value in enumerate(results):
                    actual = value[idx].float()
                    errors[arm] = max(errors[arm], (actual-ref).abs().max().item())
                    worst.append(dict(request=i, arm=arm, max_abs=(actual-ref).abs().max().item()))
            delta = (results[0].float() - results[1].float()).abs()
            diagnostic = None
            if delta.max().item() > 0.0005:
                flat_index = int(delta.argmax())
                row, remainder = divmod(flat_index, heads * dim)
                head, column = divmod(remainder, dim)
                request = next(i for i in range(len(qlens)) if starts[i] <= row < starts[i+1])
                c, ql, offset = contexts[request], qlens[request], offsets[request]
                cutoff = c - ql + row - starts[request]
                shifts = torch.arange(-64, 65)
                score = (qc[row, head].float() @ k_cpu[offset:offset+c, 0].float().T) * dim ** -.5
                shifted = score.expand(len(shifts), -1).clone()
                shifted.masked_fill_(torch.arange(c)[None, :] > (cutoff + shifts)[:, None], -float('inf'))
                references = shifted.softmax(-1) @ v_cpu[offset:offset+c, 0].float()
                distances = (references - results[1][row, head].float()).abs().amax(-1)
                best = int(distances.argmin())
                diagnostic = dict(row=row, head=head, column=column, request=request,
                    causal_cutoff=cutoff, best_cutoff_shift=int(shifts[best]),
                    best_shift_max_abs=float(distances[best]), correct_cutoff_max_abs=float(distances[64]))
            assert all(torch.isfinite(out).all() for out in results)
            print(json.dumps(dict(device=torch.npu.current_device(), block=block, query=qlens, contexts=contexts, trial=trial, cutoff_diagnostic=diagnostic, cpu_samples=worst,
                  legacy_cpu_sample_max_abs=errors[0], compressed_cpu_sample_max_abs=errors[1],
                  full_legacy_compressed_max_abs=delta.max().item(),
                  full_bitwise_equal=torch.equal(results[0].view(torch.int16), results[1].view(torch.int16)))), flush=True)
            try:
                assert max(errors) < 0.005, (block, qlens, contexts, errors)
                torch.testing.assert_close(results[1].float(), results[0].float(), atol=0.005, rtol=0.04)
            except AssertionError as error:
                failures.append(dict(query=qlens, contexts=contexts, trial=trial, reason=str(error)))
    print(json.dumps(dict(kind='gate', device=torch.npu.current_device(), failures=failures)), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
