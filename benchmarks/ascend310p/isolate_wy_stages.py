# SPDX-License-Identifier: Apache-2.0
"""独占诊断 WY 与后续算子边界；同步采样可能改变时序，不签发模型质量结论。"""
import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch_npu
from check_native_wy import compare
from native_wy import install

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--out-dir', type=Path, required=True)
a = p.parse_args()
a.out_dir.mkdir(exist_ok=False)
# First reproduce in a fresh process without the native WY extension loaded.
with (a.out_dir / 'standalone-repro.log').open('x') as log:
    subprocess.run([sys.executable, str(Path(__file__).with_name('repro_chunk_gdn_nondeterminism.py')),
                    '--out', str(a.out_dir / 'standalone-repro.jsonl')],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.set_num_threads(4)
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
m = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
original = m._compute_kernel_inputs_from_torch_wy
install(str(a.work / 'wy-extension/pr17640_wy.so'))
native = m._compute_kernel_inputs_from_torch_wy
saved = torch.load(a.work / 'diagnostic-case-1.pt', weights_only=True)
q, k, v, g, beta, state = saved['cpu_inputs']
inputs = [x.npu() for x in (q, k, v, g, beta)]
kw = dict(initial_state=state.npu(), output_final_state=True,
          head_first=False, use_qk_l2norm_in_kernel=False)
# The exact same production padding and helper on CPU provide layout references.
padded = m._pad_bthd_to_chunk(q, k, v, g, beta, m.CHUNK_SIZE)[:5]
reference = original(*padded, m.CHUNK_SIZE)
frozen = {}
first = {}

def errors(actual, expected):
    result = []
    for x, y in zip(actual, expected):
        x, y = x.cpu().float(), y.cpu().float()
        delta = x - y
        result.append(dict(max_abs=delta.abs().max().item(),
            relative_rmse=(delta.square().mean().sqrt() /
                           y.square().mean().sqrt().clamp(min=1e-30)).item(),
            finite=bool(torch.isfinite(x).all())))
    return result

def output_check(actual):
    try:
        return dict(within_tolerance=True, metrics=compare(actual, saved['reference']))
    except AssertionError as e:
        return dict(within_tolerance=False, error=str(e))

with (a.out_dir / 'stages.jsonl').open('x') as log:
    for phase in ('capture_helper_sync', 'frozen_helper_clone'):
        for repeat in range(12):
            modes = ('torch', 'native') if repeat % 2 == 0 else ('native', 'torch')
            for mode in modes:
                record = dict(phase=phase, repeat=repeat, mode=mode,
                              status='DIAGNOSTIC_ONLY')
                if phase == 'capture_helper_sync':
                    def helper(*args):
                        kernels = (original if mode == 'torch' else native)(*args)
                        cpu = [x.detach().cpu() for x in kernels]
                        record['helper_vs_cpu'] = errors(cpu, reference)
                        if mode not in first:
                            first[mode] = cpu
                            frozen[mode] = tuple(x.detach().clone() for x in kernels)
                            torch.save(cpu, a.out_dir / f'helper-first-{mode}.pt')
                        record['helper_equal_first'] = [bool(torch.equal(x, y))
                                                       for x, y in zip(cpu, first[mode])]
                        return kernels
                else:
                    def helper(*args):
                        return tuple(x.clone() for x in frozen[mode])
                m._compute_kernel_inputs_from_torch_wy = helper
                actual = m.chunk_gated_delta_rule_310(*inputs, **kw)
                record.update(output_check(actual))
                if repeat == 0 or not record['within_tolerance']:
                    torch.save([x.cpu() for x in actual],
                               a.out_dir / f'{phase}-{repeat}-{mode}.pt')
                log.write(json.dumps(record) + '\n')
                log.flush()
                print(json.dumps(record), flush=True)
