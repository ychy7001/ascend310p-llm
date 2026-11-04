# SPDX-License-Identifier: Apache-2.0
"""仅当前测试进程将FwdO改接独立maskfix，随后运行原严格全GDN门禁。"""
import torch
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.ops.load_library('/maskfix/extension/maskfix_fwd_o.so')
torch.ops._C_ascend.chunk_fwd_o = torch.ops.maskfix_fwd_o.run
from check_native_wy import main
main()

# Stress the full production call order, where fixed-address FwdO alone did not reproduce.
import json
from pathlib import Path
import subprocess
import sys
subprocess.run([sys.executable, str(Path(__file__).with_name('probe_gdn_intermediates.py')),
    '--work', '/work', '--out-dir', '/maskfix/chain-validation',
    '--maskfix-library', '/maskfix/extension/maskfix_fwd_o.so'], check=True)
summary = json.loads(Path('/maskfix/chain-validation/summary.json').read_text())
if any(r['tested'] != 24 or r['outside_tolerance_runs'] or r['raw_output_differs_final_cpu']
       or any(r['interface_variation'].values()) for r in summary.values()):
    raise RuntimeError('Maskfix full-chain numerical stress failed')
rows = [json.loads(line) for line in Path('/maskfix/chain-validation/intermediates.jsonl').read_text().splitlines()]
if any(not all(row['interfaces_equal_first']['fwd_o_raw']) for row in rows):
    raise RuntimeError('Maskfix full-chain raw output is not bitwise repeatable')
