# SPDX-License-Identifier: Apache-2.0
"""不依赖权重或原生 WY 扩展的生产 chunk-GDN 重复性最小复现。"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--repeats', type=int, default=12)
p.add_argument('--device', type=int, choices=(0, 1), default=0)
p.add_argument('--tokens', type=int, default=111)
p.add_argument('--seed', type=int, default=311)
p.add_argument('--value-heads', type=int, choices=[16, 24], default=16)
a = p.parse_args()
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.set_num_threads(4)
torch.npu.set_device(a.device)
torch_npu.npu.set_compile_mode(jit_compile=False)
m = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
assert not hasattr(m, '_production_wy_before_native'), 'Expected original production helper'
torch.manual_seed(a.seed)
q = torch.nn.functional.normalize(torch.randn(1, a.tokens, 8, 128), dim=-1).half()
k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).half()
v = (torch.randn(1, a.tokens, a.value_heads, 128) * .1).half()
g = -torch.rand(1, a.tokens, a.value_heads)
beta = (.05 + .9 * torch.rand(1, a.tokens, a.value_heads)).half()
state = (torch.randn(1, a.value_heads, 128, 128) * .01).half()
# Independent token recurrence: no chunk/WY or production FwdH/FwdO reuse.
qh, kh = (x.float().repeat_interleave(a.value_heads // 8, dim=2) for x in (q, k))
h = state[0].float().clone()
expected = torch.empty_like(v, dtype=torch.float32)
for t in range(a.tokens):
    h = h * g[0, t].exp()[:, None, None]
    delta = (v[0, t].float() - (h * kh[0, t, :, None, :]).sum(-1)) * beta[0, t, :, None].float()
    h = h + delta[:, :, None] * kh[0, t, :, None, :]
    expected[0, t] = (h * qh[0, t, :, None, :]).sum(-1) * 128 ** -.5
expected_state = h.unsqueeze(0)
inputs = [x.npu() for x in (q, k, v, g, beta)]
state_npu = state.npu()
first = None
rows = []
with a.out.open('x') as log:
    config = dict(kind='config', seed=a.seed, tokens=a.tokens, hk=8, hv=a.value_heads, dim=128,
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
                  source=str(m.__file__), scope='diagnostic, original production operators, synthetic inputs')
    log.write(json.dumps(config) + '\n')
    for repeat in range(a.repeats):
        output, final = m.chunk_gated_delta_rule_310(*inputs, initial_state=state_npu,
            output_final_state=True, head_first=False, use_qk_l2norm_in_kernel=False)
        actual = [output.cpu(), final.cpu()]
        if first is None:
            first = actual
        metrics = []
        for x, y in zip(actual, (expected, expected_state)):
            delta = x.float() - y
            bad = delta.abs() > (.0002 + .04 * y.abs())
            metrics.append(dict(max_abs=delta.abs().max().item(),
                relative_rmse=(delta.square().mean().sqrt() / y.square().mean().sqrt()).item(),
                outside_tolerance=int(bad.sum()), finite=bool(torch.isfinite(x).all())))
        bad_tokens = ((actual[0].float()-expected).abs() > (.0002+.04*expected.abs())).nonzero()
        row = dict(kind='diagnostic', repeat=repeat, metrics=metrics,
            output_equal_first=bool(torch.equal(actual[0], first[0])),
            state_equal_first=bool(torch.equal(actual[1], first[1])),
            affected_tokens=bad_tokens[:, 1].unique().tolist() if len(bad_tokens) else [])
        rows.append(row)
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps(row), flush=True)
    summary = dict(kind='summary', samples=len(rows),
        within_tolerance=sum(all(m['finite'] and m['outside_tolerance']==0 and m['relative_rmse']<=.02
                                 for m in r['metrics']) for r in rows),
        output_equal_first=sum(r['output_equal_first'] for r in rows),
        state_equal_first=sum(r['state_equal_first'] for r in rows),
        diagnostic_only=True)
    log.write(json.dumps(summary) + '\n')
    print(json.dumps(summary), flush=True)
