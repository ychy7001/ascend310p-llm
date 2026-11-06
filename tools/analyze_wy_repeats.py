#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare repeated model boundaries and independently check fixed-state H/O chunks."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import torch
from analyze_wy_boundaries import compare_samples, difference


def analyze(root, reference):
    baselines = {}
    counts = defaultdict(int)
    rows = []
    for arm in ('A', 'B'):
        for file in sorted((root / arm / 'captures').glob('*.pt')):
            r = torch.load(file, map_location='cpu', weights_only=False)
            meta = r['metadata']; request = meta['request']
            prefix = (arm, request['id'], meta['rank'], meta['layer'], meta['stage'])
            ordinal_key = (*prefix, request['repeat'])
            ordinal = counts[ordinal_key]; counts[ordinal_key] += 1
            key = (*prefix, ordinal)
            row = {'arm': arm, 'case': request['id'], 'repeat': request['repeat'], 'rank': meta['rank'],
                   'layer': meta['layer'], 'stage': meta['stage'], 'ordinal': ordinal,
                   'call': meta['call'], 'file': str(file.relative_to(root))}
            samples = {k:v for k,v in r.items() if k not in ('metadata','layer_zero_chunks','initial_state_full')}
            if key not in baselines:
                baselines[key] = (request['repeat'], samples)
            else:
                base_repeat, base_samples = baselines[key]
                row['baseline_repeat'] = base_repeat
                row['repeat_vs_baseline'] = compare_samples(samples, base_samples)
            if 'initial_state_full' in r:
                state = r['initial_state_full']
                row['incoming_state_nonzero_elements'] = int(torch.count_nonzero(state))
                row['incoming_state_max_abs'] = state.float().abs().max().item()
            checks = []
            for c in r.get('layer_zero_chunks', []):
                state = reference.decode_state_nz(c['state_nz'])
                check = {'chunk_index':c['index']}
                if meta['stage'] == 'H':
                    vn, next_state = reference.h_reference(c['k'],c['w'],c['u'],c['g'],state,actual_v_new=c['v_new'])
                    actual = c['next_state']
                    if c['next_state_layout'] == 'NZ': actual = reference.decode_state_nz(actual)
                    check['v_new_vs_CPU'] = difference(c['v_new'],vn)
                    check['next_state_given_actual_v_new_vs_CPU'] = difference(actual,next_state)
                    if c['index']==0:
                        check['h0_vs_incoming_state'] = difference(state,r['initial_state_full'])
                else:
                    expected = reference.o_reference(c['q'],c['k'],c['v_new'],c['g'],state,c['scale'])
                    check['O_given_actual_H_and_v_new_vs_CPU'] = difference(c['out'],expected)
                checks.append(check)
            if checks: row['fixed_state_chunk_checks'] = checks
            rows.append(row)
    if not rows or not all(any(x['arm']==a for x in rows) for a in ('A','B')):
        raise ValueError('Both arm captures required')
    first = {}
    for row in rows:
        changed = {k:v for k,v in row.get('repeat_vs_baseline',{}).items() if v.get('exact') is False or v.get('finite') is False or v.get('sample_layout_mismatch')}
        if not changed: continue
        key=(row['arm'],row['case'],row['repeat'],row['rank'])
        if key not in first or row['call'] < first[key]['call']:
            first[key] = {k:v for k,v in row.items() if k!='fixed_state_chunk_checks'}
            first[key]['changed_samples'] = changed
    return {'purpose':'diagnostic_only_no_model_gate_or_adoption','boundary_records':len(rows),
            'first_observed_changes':list(first.values()),'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('chunk_reference',a.reference)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    result=analyze(a.captures,ref)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'boundary_records':result['boundary_records'],'first_observed_changes':len(result['first_observed_changes'])}))


if __name__=='__main__':main()
