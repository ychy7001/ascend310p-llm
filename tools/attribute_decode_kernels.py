#!/usr/bin/env python3
"""CPU-only ordered join of measured FP16 kernels to independently observed modules.

The join is an inference constrained by 220 calls and a stable kernel-name order.
It is not a claim that an ACLGraph trace carries module names or logical shapes.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
from analyze_decode_profile import load_kernels, stable_window


def attribute(rows, ledger, lo, hi, expected_steps):
    heads=[r['start'] for r in rows if r['category']=='lm_head_fp16']
    by_step=defaultdict(list)
    for row in rows:
        if lo <= row['start'] < hi and row['category']=='fp16_nz_projection':
            by_step[bisect.bisect_right(heads,row['start'])-1].append(row)
    if len(by_step)!=expected_steps:
        raise ValueError('Incomplete selected decode intervals')
    for values in by_step.values():
        if len(values)!=len(ledger):raise ValueError('FP16 kernel/module count mismatch')
    sequences={tuple(r['name'] for r in values) for values in by_step.values()}
    if len(sequences)!=1:raise ValueError('FP16 kernel order changes between steps')
    totals=Counter();nodes=[]
    for ordinal,module in enumerate(ledger):
        values=[v[ordinal] for v in by_step.values()]
        measured=sum(v['duration'] for v in values)/expected_steps/1000
        totals[module['group']]+=measured
        nodes.append({**module,'kernel_name':values[0]['name'],'measured_ms_per_step':measured,
                      'device_identifiers':sorted({(v.get('hardware_model',v['model']),v.get('hardware_stream',v['stream']),v.get('hardware_task',v['task'])) for v in values}),
                      'kernel_shape_samples':sorted({v['shapes'] for v in values}),
                      'kernel_join_evidence':'inferred ordinal join; exact node count and stable kernel-name sequence across all selected steps'})
    return {'steps':expected_steps,'matched_fp16_calls_per_step':len(ledger),
            'join_method':'Runtime module/FX descriptors from separate B3 mapping forwards; ordered join to stable kernel sequence. No direct ACLGraph module correlation.',
            'groups_ms_per_step':dict(totals),'total_ms_per_step':sum(totals.values()),
            'validation':{'all_step_counts_match':True,'identical_kernel_name_order':True},'nodes':nodes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=96);p.add_argument('--tail',action='store_true');a=p.parse_args()
    _,rows=load_kernels(a.csv);lo,hi,count,_=stable_window(rows,a.steps,a.tail)
    ledger=next(r for r in json.loads(a.ledger.read_text()) if r['rank']==a.rank)['fp16_ledger']
    result=attribute(rows,ledger,lo,hi,count);result['rank']=a.rank
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='nodes'}))

if __name__=='__main__':main()
