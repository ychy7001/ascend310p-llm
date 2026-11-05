#!/usr/bin/env python3
"""Compare every frozen RAG/general profile using three-round medians."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


CASES = {(profile, n, c): output for profile, lengths, output in
         [('general', (1024, 4096, 16384, 32768), 256),
          ('RAG', (8192, 16384, 32768), 32)] for n in lengths for c in (1, 4)}
METRICS = {'request_prefill_tok_s': 1, 'aggregate_output_tok_s': 1,
           'median_ttft_s': -1, 'median_tpot_s': -1}


def load(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    config = events[0]
    assert config['kind'] == 'config' and config['repeats'] == 3 and config['seed'] == 17
    assert config['unit_token_ids']
    assert {tuple(row[:3]): row[3] for row in config['cases']} == CASES
    groups = defaultdict(list)
    for row in events[1:]:
        assert row['kind'] == 'round' and row['valid']
        key = (row['profile'], row['prompt_tokens'], row['concurrency'])
        assert key in CASES and row['output_tokens'] == CASES[key]
        assert row['prefix_delta']['prefix_cache_hits_total'] == 0
        assert len(row['responses']) == key[2]
        for response in row['responses']:
            assert response['usage']['prompt_tokens'] == key[1]
            assert response['usage']['completion_tokens'] == CASES[key]
        assert all(row[metric] > 0 for metric in METRICS)
        groups[key].append(row)
    assert set(groups) == set(CASES)
    for rows in groups.values():
        assert len(rows) == 3 and {row['repeat'] for row in rows} == {0, 1, 2}
    return config, groups


def adjudicate(a, b):
    ca, a = load(a)
    cb, b = load(b)
    assert ca['unit_token_ids'] == cb['unit_token_ids']
    results = []
    regressions = []
    for key in CASES:
        row = dict(profile=key[0], prompt_tokens=key[1], concurrency=key[2], metrics={})
        for metric, direction in METRICS.items():
            av = statistics.median(r[metric] for r in a[key])
            bv = statistics.median(r[metric] for r in b[key])
            gain = 100 * (bv / av - 1) if direction == 1 else 100 * (1 - bv / av)
            row['metrics'][metric] = dict(A=av, B=bv, improvement_pct=gain)
            if gain < 0:
                regressions.append(dict(profile=key, metric=metric, improvement_pct=gain))
        results.append(row)
    return dict(profiles=results, regressions=regressions,
                all_measured_medians_noninferior=not regressions,
                tolerance_added=False,
                limitations='Three rounds per shape report signed median changes; they do not establish tail latency. WY Prefill is reported separately.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--A', type=Path, required=True)
    p.add_argument('--B', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = adjudicate(args.A, args.B)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
