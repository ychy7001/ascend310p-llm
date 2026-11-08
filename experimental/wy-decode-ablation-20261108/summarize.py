#!/usr/bin/env python3
"""Summarize fixed decode windows; retain latency definitions and failed windows."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def median(values):
    return statistics.median(values) if values else None


def summarize(root):
    windows = []
    by_arm = {}
    for window in sorted(root.glob('w[0-9][0-9]-mtp*')):
        status_path = window / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'unknown'}
        freeze = json.loads((window / 'freeze.json').read_text())
        runtime_path = window / 'runtime-cli.json'
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
        record = {'window': window.name, 'status': status, 'freeze': freeze, 'runtime': runtime,
                  'rounds': []}
        for path in sorted(window.glob('c[14]-r*.json')):
            raw = json.loads(path.read_text())
            row = {key: value for key, value in raw.items()
                   if key not in ('requests', 'metrics_before', 'metrics_after')}
            row['request_timings'] = [
                {key: request.get(key) for key in ('success', 'elapsed_s', 'ttft_s', 'mean_tpot_s',
                                                  'start_offset_s', 'usage', 'error_type', 'error')}
                for request in raw['requests']]
            row['raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            record['rounds'].append(row)
            if status['state'] == 'complete':
                assert row['valid'], (window, path)
                by_arm.setdefault(freeze['arm'], {}).setdefault(row['concurrency'], []).append(row)
        windows.append(record)
    summary = {}
    for arm, groups in by_arm.items():
        summary[arm] = {}
        for concurrency, rows in groups.items():
            requests = [r for row in rows for r in row['request_timings']]
            ttfts = [r['ttft_s'] for r in requests]
            tpots = [r['mean_tpot_s'] for r in requests]
            acceptance = [r['spec_decode']['acceptance_rate'] for r in rows
                          if r.get('spec_decode') and r['spec_decode']['acceptance_rate'] is not None]
            summary[arm][str(concurrency)] = {
                'windows': sorted(set(row['window'] for row in rows)),
                'rounds': len(rows), 'requests': len(requests),
                'success': sum(r['success'] for r in requests),
                'client_ttft_s_median': median(ttfts),
                'client_tpot_ms_median': median(tpots) * 1000,
                'client_generation_tok_s_median': median([1 / t for t in tpots]),
                'server_decode_tok_s_per_request_weighted_median': median([
                    r['server_decode_tok_s_per_request_weighted'] for r in rows]),
                'e2e_aggregate_output_tok_s_median': median([r['e2e_aggregate_output_tok_s'] for r in rows]),
                'acceptance_median': median(acceptance),
                'prefix_hits_total': sum(r['metrics_delta']['vllm:prefix_cache_hits_total'] for r in rows),
                'per_window': {name: {
                    'client_tpot_ms_median': median([r['client_tpot_s_median'] for r in rows if r['window'] == name]) * 1000,
                    'e2e_aggregate_output_tok_s_median': median([r['e2e_aggregate_output_tok_s'] for r in rows if r['window'] == name])}
                    for name in sorted(set(r['window'] for r in rows))}}
    comparisons = {}
    for reference in ('0', 'reference'):
        for arm in ('1', '2'):
            for c in ('1', '4'):
                if c not in summary.get(reference, {}) or c not in summary.get(arm, {}):
                    continue
                baseline, candidate = summary[reference][c], summary[arm][c]
                comparisons[f'{arm}_vs_{reference}_c{c}'] = {
                    'client_generation_throughput_change_pct': 100 * (candidate['client_generation_tok_s_median'] / baseline['client_generation_tok_s_median'] - 1),
                    'client_tpot_reduction_pct': 100 * (1 - candidate['client_tpot_ms_median'] / baseline['client_tpot_ms_median']),
                    'e2e_aggregate_throughput_change_pct': 100 * (candidate['e2e_aggregate_output_tok_s_median'] / baseline['e2e_aggregate_output_tok_s_median'] - 1)}
    return {'protocol': json.loads((root / 'protocol.json').read_text()),
            'status': json.loads((root / 'status.json').read_text()),
            'summary': summary, 'comparisons': comparisons, 'windows': windows,
            'metric_notes': {
                'client_generation_tok_s': '1/TPOT; first nonempty text to final usage, divided by output_tokens-1; MTP may stream multiple first-event tokens, so this is a long-output client estimate',
                'server_decode_tok_s': 'output_tokens-minus-request-count divided by summed server request decode durations; per-request weighted metric, not aggregate throughput',
                'e2e_aggregate_output_tok_s': 'all output tokens / round wall time; includes cold prefill and queueing',
                'limits': 'one fixed synthetic prompt, 512 output; two boots per arm; no confidence interval or workload-wide optimum claim; reference windows sequential after mirrored MTP block'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'state': result['status'], 'summary': result['summary'],
                      'comparisons': result['comparisons']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
