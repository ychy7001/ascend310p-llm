#!/usr/bin/env python3
"""Summarize a two-rank QKVZ screen; never treat operator evidence as model adoption."""
import argparse
import json
from pathlib import Path
import statistics

POOL_FRACTION = .0752
TARGET = .03
MIN_SPEEDUP = 1 / (1 - TARGET / POOL_FRACTION)


def summarize(ranks):
    assert sorted(r['rank'] for r in ranks) == [0, 1]
    assert len({r['manifest_sha256'] for r in ranks}) == 1
    strata = []
    for rank in ranks:
        assert rank['g0_passed'] and len(rank['g0']) == 120
        for batch in (1, 2, 3, 4):
            rows = [t for t in rank['timings'] if t['batch'] == batch]
            assert len(rows) == 3 and [r['order'] for r in rows] == ['AB', 'BA', 'AB']
            assert all(r['steps'] == 256 for r in rows)
            speedup = statistics.median(r['speedup'] for r in rows)
            strata.append({'rank': rank['rank'], 'batch': batch, 'mode': rows[0]['mode'],
                           'A_ms_30': statistics.median(r['ms_per_30_projections']['A'] for r in rows),
                           'B_ms_30': statistics.median(r['ms_per_30_projections']['B'] for r in rows),
                           'median_paired_speedup': speedup,
                           'decode_only_budget_reduction_fraction': POOL_FRACTION*(1-1/speedup),
                           'clears_ideal_budget_threshold': speedup >= MIN_SPEEDUP})
    return {'status': 'screen_pass_requires_model_gates' if all(s['clears_ideal_budget_threshold'] for s in strata)
            else 'stop_screen_below_budget',
            'ideal_min_speedup': MIN_SPEEDUP, 'pure_decode_pool_fraction': POOL_FRACTION,
            'target_replay_reduction_fraction': TARGET, 'strata': strata,
            'G0': {'cases': sum(len(r['g0']) for r in ranks), 'passed': True,
                   'max_nrmse': max(g['nrmse'] for r in ranks for g in r['g0']),
                   'min_cosine': min(g['cosine'] for r in ranks for g in r['g0'])},
            'G1': 'not_run', 'G2': 'not_run', 'model_B_runs': 0,
            'limits': ['real weights, synthetic activations', 'ranks measured sequentially without TP2 communication',
                       'contexts 4K/16K/32K/80K/94K not sampled',
                       'no measured meeting/RAG/general end-to-end improvement',
                       'per-stratum budget is a scenario using the overall pool fraction, not a workload estimate',
                       'no WY change or WY Prefill measurement']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranks', type=Path, nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize([json.loads(path.read_text()) for path in a.ranks])
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
