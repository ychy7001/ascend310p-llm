#!/usr/bin/env python3
"""Check dual-rank MoE kernels in native stable Decode profile windows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from analyze_decode_profile import analyze


def observations(folder):
    events = [json.loads(line) for line in (folder / 'profile-events.jsonl').read_text().splitlines()]
    windows = {}
    for batch in range(1, 5):
        start = next(e for e in events if e['kind'] == 'profile_start' and e['batch'] == batch)
        stop = next(e for e in events if e['kind'] == 'profile_stop' and e['batch'] == batch)
        assert len(stop['stable_steps']) == 2
        assert all(64 <= n <= 128 for n in stop['stable_steps'].values())
        windows[batch] = (start['mono'], stop['mono'])
    modes = Counter()
    for path in sorted((folder / 'observation').glob('*.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] != 'forward_start':
                    continue
                for batch, (start, stop) in windows.items():
                    if start <= row['mono'] <= stop:
                        assert row['batch'] == row['padded_tokens'] == batch
                        assert row['graph_hit'] == (batch <= 2)
                        assert row['mode'] == ('FULL' if batch <= 2 else 'NONE')
                        assert not row['capturing']
                        modes[(row['rank'], batch)] += 1
    assert set(modes) == {(rank, batch) for rank in (0, 1) for batch in range(1, 5)}
    assert all(64 <= n <= 128 for n in modes.values())
    return [{'rank': rank, 'batch': batch, 'stable_observed_steps': n,
             'graph': batch <= 2} for (rank, batch), n in sorted(modes.items())]


def adjudicate(root):
    records = []
    checks = {}
    for arm in ('A', 'B'):
        folder = root / ('profile-' + arm)
        checks[arm] = observations(folder)
        for rank in (0, 1):
            captures = sorted(p for p in folder.glob('device-profiles/**/PROF*')
                              if p.is_dir() and re.search(r'_rank' + str(rank) + r'_', str(p.parent)))
            assert len(captures) == 4, (arm, rank, len(captures))
            for batch, capture in enumerate(captures, 1):
                devices = list(capture.glob('device_*'))
                assert len(devices) == 1
                csvs = list(devices[0].glob('summary/op_summary*.csv'))
                assert len(csvs) == 1
                result, rows = analyze(csvs[0], None, 96)
                assert result['steps'] == 96
                candidate_name = 'moe_prescale_0' if batch <= 2 else 'moe_prescale'
                kernels = [r for r in rows if 'QuantGroupedMatmulDequant' in r['name']
                           or r['name'] == candidate_name]
                counts = Counter(r['name'] for r in kernels)
                assert len(kernels) == 80 * 96, (arm, rank, batch, dict(counts))
                if arm == 'A':
                    assert all('QuantGroupedMatmulDequant' in r['name'] for r in kernels)
                else:
                    assert all(r['name'] == candidate_name for r in kernels)
                lo, hi = result['window_start_us'], result['window_end_us']
                cost = sum(min(hi, r['end']) - max(lo, r['start']) for r in kernels) / 96 / 1000
                records.append(dict(arm=arm, rank=rank, batch=batch, graph=batch <= 2,
                                    steps=96, moe_kernel_counts=dict(counts), moe_ms_per_step=cost,
                                    window_ms_per_step=result['window_ms_per_step'],
                                    uncovered_by_named_kernel_ms_per_step=result['uncovered_by_named_kernel_ms_per_step']))
    return dict(passed=True, captures=len(records), stable_steps_per_capture=96,
                observations=checks, records=records,
                limitation='Profiling explains kernel execution; adoption uses separate unprofiled replay.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = adjudicate(args.root)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
