# SPDX-License-Identifier: Apache-2.0
"""默认GDN多形状门禁；复用独立CPU递推，不安装任何WY/H/O补丁。"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import torch
import torch_npu
from check_native_wy import compare, recurrent_reference

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--device', type=int, choices=(0, 1), default=0)
a = p.parse_args()
torch.set_num_threads(4)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.npu.set_device(a.device)
torch_npu.npu.set_compile_mode(jit_compile=False)
m = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
assert not hasattr(m, '_production_wy_before_native')
helper = m._compute_kernel_inputs_from_torch_wy
cases = [(t, 8, 16, 1., None) for t in (1, 63, 64, 65, 111, 127, 128, 129, 512, 1025)]
cases += [(128, 16, 32, 1., None), (177, 8, 16, 1., [0, 1, 65, 177]),
          (128, 8, 24, 1., None), (512, 8, 16, 8., None), (64, 8, 16, .01, None)]
rows = []
with a.out.open('x') as log:
    log.write(json.dumps(dict(kind='config', chunk_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
                             cases=len(cases), atol=.0002, rtol=.04, max_relative_rmse=.02,
                             independent_reference=True, experimental_patches=False)) + '\n')
    for idx, (tokens, hk, hv, decay, cu) in enumerate(cases):
        torch.manual_seed(310 + idx)
        q = torch.nn.functional.normalize(torch.randn(1, tokens, hk, 128), dim=-1).half()
        k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).half()
        v = (torch.randn(1, tokens, hv, 128) * .1).half()
        g = -torch.rand(1, tokens, hv) * decay
        beta = (.05 + .9 * torch.rand(1, tokens, hv)).half()
        cu = cu or [0, tokens]
        state = (torch.randn(len(cu) - 1, hv, 128, 128) * .01).half()
        expected = recurrent_reference(q, k, v, g, beta, state, cu)
        device = [x.npu() for x in (q, k, v, g, beta)]
        kw = dict(initial_state=state.npu(), output_final_state=True,
                  head_first=False, use_qk_l2norm_in_kernel=False)
        if len(cu) > 2:
            kw['cu_seqlens'] = torch.tensor(cu, dtype=torch.int64, device='npu')
        row = dict(kind='case', case=idx, tokens=tokens, hk=hk, hv=hv, decay=decay, cu=cu,
                   nonzero_initial_state=True)
        try:
            actual = m.chunk_gated_delta_rule_310(*device, **kw)
            row['vs_independent_reference'] = compare(actual, expected)
            for _ in range(3):
                compare(m.chunk_gated_delta_rule_310(*device, **kw), actual, atol=0, rtol=0)
            assert m._compute_kernel_inputs_from_torch_wy is helper
            row.update(status='PASS', deterministic_repeats=3)
        except AssertionError as error:
            row.update(status='FAIL', error=str(error))
        rows.append(row)
        log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
    summary = dict(kind='summary', cases=len(rows), passed=sum(r['status'] == 'PASS' for r in rows),
                   scope='Default operator gate; no model/performance acceptance')
    log.write(json.dumps(summary) + '\n'); print(json.dumps(summary), flush=True)
if summary['passed'] != summary['cases']:
    raise SystemExit(1)
