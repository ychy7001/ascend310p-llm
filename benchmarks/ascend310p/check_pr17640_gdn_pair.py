# SPDX-License-Identifier: Apache-2.0
"""成对替换当前诊断进程的 H/O，沿用独立 CPU 参考及原生 WY 严格门禁。"""
import argparse
import json
from pathlib import Path
import sys
import torch
from vllm_ascend.utils import enable_custom_op

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', required=True)
p.add_argument('--wy-library', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert enable_custom_op()
torch.ops.load_library(a.library)
torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h = torch.ops.pr17640_gdn_pair.fwd_h
torch.ops._C_ascend.chunk_fwd_o = torch.ops.pr17640_gdn_pair.fwd_o
sys.argv = ['check_native_wy.py', '--library', a.wy_library, '--out', str(a.out)]
from check_native_wy import main
main()
rows = [json.loads(line) for line in a.out.read_text().splitlines()]
assert len(rows) == 10 and all(row['status'] == 'PASS' for row in rows), rows
# Torch-WY + paired H/O 也必须独立通过，不能仅候选通过就隐去原路径错误。
assert all(not row.get('baseline_reference_error') for row in rows), rows
