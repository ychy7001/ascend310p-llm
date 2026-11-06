#!/usr/bin/env python3
"""Apply frozen five-pair MoE replay adoption rules to completed observations."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

PLAN_SHA256 = '975945e7aee37ad1a0ae3723187f59050ff892e1c9c21ecd2b9e1c2a8959253e'
ORDERS = [('A','B'),('B','A'),('A','B'),('B','A'),('A','B')]


def summarize(root, label):
    folder = root / label
    events = [json.loads(line) for line in (folder/'http-events.jsonl').read_text().splitlines()]
    config = next(e for e in events if e['kind']=='config')
    assert config['plan_sha256']==PLAN_SHA256 and config['seed']==17
    begin = next(e for e in events if e['kind']=='replay_start')
    end = next(e for e in events if e['kind']=='replay_end')
    requests = [e for e in events if e['kind']=='request_end']
    starts = {e['label']:e for e in events if e['kind']=='request_start'}
    assert len(requests)==len(starts)==end['requests']==11
    assert sum(r['usage']['completion_tokens'] for r in requests)==end['output_tokens']==17045
    assert sum(r['usage']['prompt_tokens'] for r in requests)==end['input_tokens']==200895
    assert end['prefix']['prefix_cache_hits_total']==begin['prefix']['prefix_cache_hits_total']
    assert not any(e['kind']=='request_error' for e in events)
    stage = json.loads((folder/'analysis/replay-summary.json').read_text())
    validation = stage['validation']
    for key in ('all_requests_identified','request_count_matches','http_server_token_counts_match',
                'no_negative_computed','prompt_tokens_accounted','prefix_zero_hits'):
        assert validation[key], (label,key)
    assert not validation['unmatched_replay_worker_sequences']
    assert abs(validation['wall_partition_residual_s'])<1e-6
    assert stage['requests']==11 and stage['output_tokens']==17045
    for row in requests:
        row['prompt_tokens'] = row['usage']['prompt_tokens']
        row['concurrency'] = sum(other['start']<=row['first']<other['end'] for other in requests)
    return dict(wall_s=end['mono']-begin['mono'],requests=requests,
                phase_seconds=stage['phase_seconds'],plan_sha256=config['plan_sha256'],
                seed=config['seed'],unit_token_ids=config['unit_token_ids'],
                max_arrival_lateness_s=max(r['arrival_lateness_seconds'] for r in starts.values()))


def validate_summary(row):
    for value in [row['wall_s'], *(r[k] for r in row['requests'] for k in ('ttft_s', 'mean_tpot_s'))]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Replay timings must be finite positive numbers')
    if len(row['requests']) != 11 or len({r['label'] for r in row['requests']}) != 11:
        raise ValueError('Expected eleven unique replay requests')


def adjudicate(root):
    pairs = []
    arms = {arm:[] for arm in ('A','B')}
    identity = None
    for i, order in enumerate(ORDERS,1):
        rows = {arm:summarize(root,f'p{i}-{arm}') for arm in order}
        for arm,row in rows.items():
            validate_summary(row)
            current = (row['plan_sha256'],row['seed'],row['unit_token_ids'])
            if identity is None:
                identity = current
            assert current==identity, 'Frozen input identity changed'
            arms[arm].append(row)
        pairs.append(dict(pair=i,order=order,A_wall_s=rows['A']['wall_s'],B_wall_s=rows['B']['wall_s'],
                          reduction_pct=100*(1-rows['B']['wall_s']/rows['A']['wall_s'])))
    # Request IDs are fixed; group both by prompt length and by observed concurrency.
    strata = defaultdict(lambda:defaultdict(list))
    for arm,runs in arms.items():
        for run in runs:
            for row in run['requests']:
                n = row['prompt_tokens']
                bucket = next(name for ceiling,name in [(8192,'0-8K'),(16384,'8-16K'),
                             (32768,'16-32K'),(65536,'32-64K')] if n<ceiling)
                for key in ('request:'+row['label'],'context:'+bucket,'concurrency:C'+str(row['concurrency'])):
                    strata[key][arm].append(row)
    comparisons = []
    for name,groups in sorted(strata.items()):
        counts = {arm:len(groups[arm]) for arm in ('A','B')}
        row = dict(stratum=name,counts=counts,sufficient=min(counts.values())>=5)
        for metric in ('ttft_s','mean_tpot_s'):
            values = {arm:statistics.median(r[metric] for r in groups[arm]) if groups[arm] else None
                      for arm in ('A','B')}
            row[metric] = dict(medians=values,regression_pct=
                100*(values['B']/values['A']-1) if min(counts.values()) else None)
        comparisons.append(row)
    median = statistics.median(p['reduction_pct'] for p in pairs)
    faster = sum(p['reduction_pct']>0 for p in pairs)
    regressions = [r['stratum']+':'+metric for r in comparisons if r['sufficient']
                   for metric in ('ttft_s','mean_tpot_s') if r[metric]['regression_pct']>3]
    return dict(pairs=pairs,median_wall_reduction_pct=median,faster_pairs=faster,
                strata=comparisons,stratum_regressions_over_3pct=regressions,
                replay_performance_pass=median>=3 and faster>=4 and not regressions,
                load_profiles_required_separately=True,semantic_review_required_before_adoption=True,
                phase_seconds={arm:[r['phase_seconds'] for r in runs] for arm,runs in arms.items()},
                max_arrival_lateness_s=max(r['max_arrival_lateness_s'] for runs in arms.values() for r in runs),
                limitations='Observed concurrency can change with service speed; sparse strata are reported separately. Five pairs do not establish tail-latency improvements.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    result = adjudicate(args.root)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
