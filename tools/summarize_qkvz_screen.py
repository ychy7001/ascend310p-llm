#!/usr/bin/env python3
"""Apply frozen prior QKVZ pool weights to paired full-path operator measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(ranks, manifest, budget):
    assert sorted(r['rank'] for r in ranks) == [0, 1]
    assert len({r['manifest_sha256'] for r in ranks}) == 1
    assert len(manifest['rows']) == 60
    assert all(r['zero_int8_exact'] and r['zero_scale_exact'] for r in manifest['rows'])
    pool = budget['pool_fraction']
    assert budget['target'] == .03 and pool > budget['target']
    weights = {(r['rank'], r['batch']): r['replay_wall_pool_fraction'] for r in budget['strata']}
    assert set(weights) == {(r, b) for r in (0, 1) for b in (1, 2, 3, 4)}
    assert abs(sum(weights.values())-pool) < 1e-12 and all(v >= 0 for v in weights.values())
    strata = []
    for rank in ranks:
        assert rank['action'] == 'bench' and not rank['g0']
        for batch in (1, 2, 3, 4):
            rows = [t for t in rank['timings'] if t['batch'] == batch]
            assert len(rows) == 3 and [r['order'] for r in rows] == ['AB', 'BA', 'AB']
            assert [r['round'] for r in rows] == [0, 1, 2]
            assert all(r['steps'] == 256 for r in rows)
            assert all(math.isfinite(r['speedup']) and r['speedup'] > 0 for r in rows)
            assert all(abs(r['speedup'] - r['ms_per_30_projections']['A']/r['ms_per_30_projections']['B']) < 1e-12 for r in rows)
            speedup = statistics.median(r['speedup'] for r in rows)
            fraction = weights[rank['rank'], batch]
            strata.append({'rank': rank['rank'], 'batch': batch, 'mode': rows[0]['mode'],
                           'A_ms_30': statistics.median(r['ms_per_30_projections']['A'] for r in rows),
                           'B_ms_30': statistics.median(r['ms_per_30_projections']['B'] for r in rows),
                           'median_paired_speedup': speedup,
                           'prior_replay_wall_pool_fraction': fraction,
                           'budget_reduction_fraction': fraction*(1-1/speedup)})
    reduction = sum(s['budget_reduction_fraction'] for s in strata)
    return {'status': 'screen_pass_requires_model_gates' if reduction >= budget['target'] else 'stop_screen_below_budget',
            'ideal_min_speedup': 1/(1-budget['target']/pool),
            'pure_decode_pool_fraction': pool,
            'weighted_effective_speedup': pool / sum(s['prior_replay_wall_pool_fraction']/s['median_paired_speedup'] for s in strata),
            'projected_decode_only_replay_reduction_fraction': reduction,
            'target_replay_reduction_fraction': budget['target'], 'strata': strata,
            'quantization_policy': manifest['policy'],
            'zero_channels_by_layer_rank': [
                {k: r[k] for k in ('module', 'rank', 'fp16_zero_channels',
                                   'bf16_to_fp16_underflow_channels', 'bf16_underflow_absmax_max',
                                   'zero_int8_exact', 'zero_scale_exact')}
                for r in manifest['rows']],
            'bf16_to_fp16_underflow_channels_total': manifest['bf16_to_fp16_underflow_channels_total'],
            'bf16_underflow_absmax_max': manifest['bf16_underflow_absmax_max'],
            'G0': 'not_run', 'G1': 'not_run', 'G2': 'not_run', 'model_B_runs': 0,
            'limits': ['real weights, synthetic activations', 'ranks measured sequentially without TP2 communication',
                       'contexts 4K/16K/32K/80K/94K not sampled',
                       'no measured meeting/RAG/general end-to-end improvement',
                       'projected reduction transfers prior profile kernel pools; not an end-to-end measurement',
                       'no WY change or WY Prefill measurement']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranks', type=Path, nargs=2, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--budget', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ranks = [json.loads(path.read_text()) for path in a.ranks]
    manifest_sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    assert all(r['manifest_sha256'] == manifest_sha for r in ranks)
    result = summarize(ranks, json.loads(a.manifest.read_text()), json.loads(a.budget.read_text()))
    result['manifest_sha256'] = manifest_sha
    result['budget_sha256'] = hashlib.sha256(a.budget.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'zero_channels_by_layer_rank'}, indent=2))


if __name__ == '__main__': main()
