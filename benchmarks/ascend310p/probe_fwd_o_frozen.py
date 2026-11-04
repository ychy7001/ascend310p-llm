# SPDX-License-Identifier: Apache-2.0
"""固定FwdO全部输入重复调用；可在独立namespace验证maskfix，未使用模型权重。"""
import argparse
import importlib
import json
from pathlib import Path

import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--out-dir', type=Path, required=True)
p.add_argument('--maskfix-library', '--fwd-o-library', dest='maskfix_library', type=Path)
p.add_argument('--fwd-o-namespace', default='maskfix_fwd_o')
p.add_argument('--candidate-label', default='maskfix')
p.add_argument('--repeats', type=int, default=24)
a = p.parse_args()
a.out_dir.mkdir(exist_ok=False)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.set_num_threads(4)
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
m = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
saved = torch.load(a.work / 'diagnostic-case-1.pt', weights_only=True)
q, k, v, g, beta, state = saved['cpu_inputs']
padded = m._pad_bthd_to_chunk(q, k, v, g, beta, m.CHUNK_SIZE)[:5]
# Compute kernel-layout inputs on CPU once; no NPU WY execution or monkeypatch.
qk, kk, wk, uk, gk = (x.npu() for x in m._compute_kernel_inputs_from_torch_wy(*padded, m.CHUNK_SIZE))
statek = state.npu().transpose(-1, -2).contiguous()
h, vn, fs = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
    kk, wk, uk, g=gk, gk=None, initial_state=statek, output_final_state=True,
    chunk_size=64, save_new_value=True, cu_seqlens=None, chunk_indices=None,
    use_exp2=False, transpose_state_layout=False)
# Synchronize and freeze every tensor argument before repeating FwdO only.
args = tuple(x.clone() for x in (qk, kk, vn, h, gk))
torch.save([x.cpu() for x in args], a.out_dir / 'frozen-fwd-o-inputs.pt')
expected = saved['reference'][0]
ops = {'production': torch.ops._C_ascend.chunk_fwd_o}
if a.maskfix_library:
    torch.ops.load_library(str(a.maskfix_library))
    ops[a.candidate_label] = getattr(torch.ops, a.fwd_o_namespace).run
first = {}
rows = []
with (a.out_dir / 'fwd-o.jsonl').open('x') as log:
    for repeat in range(a.repeats):
        for mode in (list(ops) if repeat % 2 == 0 else list(ops)[::-1]):
            qk, kk, vn, h, gk = args
            output = ops[mode](qk, kk, vn, h, 128 ** -.5, g=gk,
                g_gamma=None, cu_seqlens=None, chunk_indices=None,
                chunk_size=64, transpose_state_layout=False)
            actual = output.transpose(1, 2)[:, :111].contiguous().cpu()
            if mode not in first:
                first[mode] = actual
            delta = actual.float() - expected
            bad = delta.abs() > (.0002 + .04 * expected.abs())
            where = bad.nonzero()
            rel = (delta.square().mean().sqrt() / expected.square().mean().sqrt()).item()
            row = dict(repeat=repeat, mode=mode, max_abs=delta.abs().max().item(),
                relative_rmse=rel, outside_tolerance=int(bad.sum()),
                affected_tokens=where[:, 1].unique().tolist() if len(where) else [],
                output_equal_first=bool(torch.equal(actual, first[mode])),
                within_tolerance=bool(torch.isfinite(actual).all()) and not bool(bad.any()) and rel <= .02,
                status='DIAGNOSTIC_ONLY', scope='identical fixed inputs; FwdO repeated alone')
            rows.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
    summary = {mode:dict(tested=sum(r['mode']==mode for r in rows),
        within_tolerance=sum(r['mode']==mode and r['within_tolerance'] for r in rows),
        output_equal_first=sum(r['mode']==mode and r['output_equal_first'] for r in rows)) for mode in ops}
    (a.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
