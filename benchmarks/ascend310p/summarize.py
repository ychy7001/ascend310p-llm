#!/usr/bin/env python3
"""汇总已有JSONL；分位数采用nearest-rank，少量样本显式标记筛选。"""
import argparse
import json
import math
from pathlib import Path


def distribution(values):
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    def q(p):
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return {'n': len(ordered), 'min': ordered[0], 'p50': q(.5),
            'p95': q(.95), 'max': ordered[-1],
            'scope': 'screening; insufficient samples for tail characterization'
            if len(ordered) < 20 else 'empirical nearest-rank; not a confidence bound'}


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in records if r.get('kind') == 'request']
    successful = [r for r in rows if r.get('success', r.get('valid', True))]
    rounds = [r for r in records if r.get('kind') == 'round']
    result = {'file': str(path), 'requests': len(rows), 'successful': len(successful),
              'failed': len(rows) - len(successful),
              'success_rate': len(successful) / len(rows) if rows else None,
              'quantile_method': 'nearest-rank; warmup excluded'}
    for key in ('prefill_tok_s', 'decode_tok_s'):
        result[key] = distribution([r.get(key) for r in successful])
    for key in ('ttft_s', 'mean_tpot_s', 'elapsed_s'):
        result[key] = distribution([r.get('response', r).get(key) for r in successful])
    result['aggregate_output_tok_s'] = distribution([r['aggregate_output_tok_s'] for r in rounds])
    spec = [r['spec_decode'] for r in successful if 'spec_decode' in r]
    drafted = sum(r['drafted_tokens'] for r in spec)
    steps = sum(r['drafts'] for r in spec)
    accepted = sum(r['accepted_tokens'] for r in spec)
    result['spec_decode'] = {'drafts': steps, 'drafted_tokens': drafted, 'accepted_tokens': accepted,
                             'acceptance_rate': accepted / drafted if drafted else None,
                             'mean_acceptance_length': 1 + accepted / steps if steps else None}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    text = json.dumps([summarize(p) for p in args.files], ensure_ascii=False, indent=2) + '\n'
    if args.out:
        with args.out.open('x') as output:
            output.write(text)
    print(text, end='')
