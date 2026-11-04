#!/usr/bin/env python3
"""采样npu-smi原始输出；峰值仅为采样峰值，不冒充精确分配器峰值。"""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-dir', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--out', type=Path)
args = parser.parse_args()
root = args.experiment_dir
with (args.out or root / 'resource-samples.jsonl').open('x') as output:
    for _ in range(1440):
        started = time.time()
        r = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
        try:
            status = json.loads((root / 'status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            status = {}
        row = {'unix_s': started, 'query_elapsed_s': time.time() - started,
               'returncode': r.returncode, 'npu_smi': r.stdout, 'stderr': r.stderr,
               'stage': status.get('current', status.get('state'))}
        output.write(json.dumps(row) + '\n')
        output.flush()
        if status.get('state') in ('complete', 'failed', 'not_started'):
            break
        time.sleep(5)
