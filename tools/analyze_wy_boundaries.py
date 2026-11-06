#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze trusted diagnostic captures with the kernel repository's CPU oracle.

The sampled diagnostics localize errors; their thresholds do not replace model G1.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import torch


def difference(actual, expected):
    if actual.shape != expected.shape:
        return {'shape_mismatch': [list(actual.shape), list(expected.shape)]}
    a, e = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    if not finite:
        return {'finite': False}
    delta = a - e
    return {'finite': True, 'elements': a.numel(), 'max_abs': delta.abs().max().item(),
            'relative_rmse': (delta.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-30)).item(),
            'exact': bool(torch.equal(a, e))}


def compare_samples(a, b, path=''):
    if isinstance(a, dict) and 'data' in a:
        if not isinstance(b, dict) or a['shape'] != b['shape'] or a['starts'] != b['starts']:
            return {path: {'sample_layout_mismatch': True}}
        return {path: difference(a['data'], b['data'])}
    rows = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() & b.keys():
            rows.update(compare_samples(a[k], b[k], path + '/' + str(k)))
    elif isinstance(a, list) and isinstance(b, list):
        for i, (aa, bb) in enumerate(zip(a, b, strict=True)):
            rows.update(compare_samples(aa, bb, path + '/' + str(i)))
    return rows


def analyze(root, oracle):
    rows, paired, counts = [], {}, defaultdict(int)
    for arm in ('A', 'B'):
        files = sorted((root / arm / 'captures').glob('*.pt'))
        if not files:
            raise ValueError('No captures for ' + arm)
        for file in files:
            record = torch.load(file, map_location='cpu', weights_only=False)
            meta = record['metadata']
            req = meta['request']
            key = (req['id'], req['position'], req['context_sha256'], meta['rank'], meta['layer'], meta['stage'])
            ordinal = counts[(arm, key)]
            counts[(arm, key)] += 1
            key += (ordinal,)
            row = {'arm': arm, 'case': req['id'], 'position': req['position'], 'rank': meta['rank'],
                   'layer': meta['layer'], 'stage': meta['stage'], 'ordinal': ordinal,
                   'file': str(file.relative_to(root)), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
            if meta['stage'] == 'WY':
                reference = oracle.reference(record['inputs'], record['chunk_size'])
                row['actual_vs_cpu'] = oracle.metrics(record['actual'], reference)
                if record['same_input_torch'] is not None:
                    row['same_input_torch_vs_cpu'] = oracle.metrics(record['same_input_torch'], reference)
                    row['actual_vs_same_input_torch'] = [difference(a, b) for a, b in zip(record['actual'], record['same_input_torch'], strict=True)]
                row['chunk_indices'] = record['chunk_indices']
                row['full_shapes'] = record['full_shapes']
                sample_record = {'inputs': [{'shape': list(x.shape), 'starts': [0], 'data': x} for x in record['inputs']],
                                 'outputs': [{'shape': list(x.shape), 'starts': [0], 'data': x} for x in record['actual']]}
            else:
                sample_record = {k: v for k, v in record.items() if k != 'metadata'}
            if arm == 'A':
                paired[key] = sample_record
            else:
                if key not in paired:
                    raise ValueError('B capture has no matching A: ' + str(key))
                row['B_vs_A_samples'] = compare_samples(sample_record, paired.pop(key))
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('arm', 'case', 'rank', 'layer', 'stage', 'ordinal')}), flush=True)
    if paired:
        raise ValueError('Unmatched A boundaries: ' + str(len(paired)))
    bad = [r for r in rows if r['stage'] == 'WY' and not all(m['passed'] for m in r['actual_vs_cpu'])]
    return {'purpose': 'diagnostic_only_no_model_gate_or_adoption', 'boundary_records': len(rows),
            'WY_CPU_tolerance_failures': len(bad), 'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--oracle-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    path = args.oracle_dir / 'wy_snapshot_reference.py'
    spec = importlib.util.spec_from_file_location('wy_oracle', path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    result = analyze(args.captures, oracle)
    result['oracle_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
