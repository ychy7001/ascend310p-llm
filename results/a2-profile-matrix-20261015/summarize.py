#!/usr/bin/env python3
"""汇总固定输入参数矩阵；保留逐次值，不把两个样本当作P95。"""
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
summary = {}
for folder in sorted(root.glob('mtp*-b*')):
    cells = {}
    for path in sorted(folder.glob('p*-o128.jsonl')):
        requests = [r for line in path.read_text().splitlines()
                    if (r := json.loads(line)).get('kind') == 'request']
        if not requests:
            continue
        cells[path.stem] = {
            'count': len(requests), 'all_valid': all(r['valid'] for r in requests),
            'prefill_tok_s': [r['prefill_tok_s'] for r in requests],
            'decode_tok_s': [r['decode_tok_s'] for r in requests],
            'prefill_median': statistics.median(r['prefill_tok_s'] for r in requests),
            'decode_median': statistics.median(r['decode_tok_s'] for r in requests),
            'acceptance': [r['spec_decode'] for r in requests],
            'prompt_hashes': [r['prompt_ids_sha256'] for r in requests],
            'output_hashes': [hashlib.sha256(r['response']['text'].encode()).hexdigest() for r in requests],
            'ttft_seconds': [r['response']['ttft_s'] for r in requests],
            'mean_tpot_seconds': [r['response']['mean_tpot_s'] for r in requests],
        }
    summary[folder.name] = cells
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
