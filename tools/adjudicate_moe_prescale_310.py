#!/usr/bin/env python3
"""Screen real grouped-MoE microbenchmarks against a frozen decode attribution.

This is an entry screen, not end-to-end acceptance. Pool weights follow the
archived per-batch, per-rank operator fractions and pure-decode time weights.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def adjudicate(rows, attribution):
    composition=json.loads((attribution/'weighted-composition.json').read_text())
    with (attribution/'per-case-composition.csv').open() as stream:
        fractions={(int(r['batch']),int(r['rank'])):float(r['fraction_of_case_cycle'])
                   for r in csv.DictReader(stream) if r['category']=='moe_grouped_w8a8'}
    batch_weights={int(re.search(r'(?:b|B)([1-4])',k)[1]):v for k,v in composition['time_weights'].items()}
    pools={b:composition['decode_wall_fraction']*batch_weights[b]*statistics.mean(fractions[b,r] for r in (0,1)) for b in (1,2,3,4)}
    assert abs(sum(pools.values())-.16163336554424895)<1e-10
    by_batch={}
    bits=True
    g0=True
    for row in rows:
        assert row.get('graph_replay_verified') is True, 'Graph input-mutation replay verification is required'
        bits &= all(row['parity'].values()) and all(row['repeat'])
        g0 &= row['parity']['compiled_baseline'] and all(row['repeat'])
        g0 &= all(x['finite'] and x['outside_tolerance']==0 for x in row['cpu_fp32'].values())
    for batch in (1,2,3,4):
        selected=[r for r in rows if r['m']==batch*8]
        mode='graph' if batch<=2 else 'eager'
        banks={re.match(r'input-(\d+)-',r['label'])[1] for r in selected}
        assert len(banks)==2, f'Expected two independent rank banks for batch {batch}'
        rank_ratios=[]
        bank_summaries=[]
        for bank in sorted(banks):
            times={'A':0.,'B':0.}
            samples={}
            for kind,k in (('up',2048),('down',256)):
                subset=[r for r in selected if r['k']==k and r['label'].startswith(f'input-{bank}-')]
                assert len(subset)>=9, f'Need 3 requests x 3 stable input snapshots per bank/K/M, got {len(subset)}'
                assert len({re.search(r'trial(\d+)',r['label'])[1] for r in subset})==3
                samples[kind]=len(subset)
                for arm in times:
                    times[arm]+=statistics.median(r[mode]['median_us'][arm] for r in subset)
            rank_ratios.append(times['B']/times['A'])
            bank_summaries.append({'samples':samples,'median_projection_pair_us':times,'speedup':times['A']/times['B']})
        ratio=statistics.mean(rank_ratios)
        by_batch[str(batch)]={'mode':mode,'rank_banks':bank_summaries,'cost_ratio_B_over_A':ratio,
                             'speedup':1/ratio,'pool_fraction_of_replay':pools[batch],
                             'projected_replay_reduction_pct':pools[batch]*(1-ratio)*100}
    improvement=sum(x['projected_replay_reduction_pct'] for x in by_batch.values())
    return {'operator_cases':len(rows),'bitwise_pass':bool(bits),'G0_pass':bool(g0),
            'G0_tolerance':{'rtol':.002,'atol':.002},'by_batch':by_batch,
            'projected_replay_reduction_pct':improvement,'entry_threshold_pct':3.,
            'enter_model_gate':bool(g0 and improvement>=3.),'requires_G1':not bool(bits),
            'performance_acceptance':False,'G1':'not run','G2':'not run',
            'profiles':{'meeting_replay':'projection only; no measured end-to-end gain',
                        'RAG_8K_32K_prefill':'not measured',
                        'general_1K_32K_C1_C4':'not measured','WY_prefill':'not measured; unchanged operator'},
            'limits':['Only the first encountered real expert layer is replayed per rank; routing is observed across all layers.',
                      'Capture is eager at 4K context; it is not the production Graph/AOT performance baseline.',
                      'The transfer estimates pure decode only; mixed/prefill stages, arrival dependence, and integration overhead are unmeasured.',
                      'Three paired timing samples do not characterize tail latency.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--micro',required=True,type=Path)
    p.add_argument('--attribution',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    result=adjudicate(json.loads(a.micro.read_text()),a.attribution)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('operator_cases','G0_pass','projected_replay_reduction_pct','enter_model_gate')},ensure_ascii=False))
