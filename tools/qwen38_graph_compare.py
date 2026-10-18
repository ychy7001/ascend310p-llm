# SPDX-License-Identifier: Apache-2.0
"""Compare preserved HTTP observations without implying model quality parity."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    config = next(row for row in rows if row['kind'] == 'config')
    measured = [row for row in rows if row['kind'] == 'request']
    assert measured
    assert all(row['usage']['prompt_tokens'] == config['prompt_tokens'] and
               row['usage']['completion_tokens'] == config['output_tokens']
               for row in measured)
    def stats(values):
        return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
    return {
        'file': str(path), 'timed_requests': len(measured),
        'prompt_tokens': config['prompt_tokens'], 'output_tokens': config['output_tokens'],
        'prompt_ids_sha256': hashlib.sha256(json.dumps(config['prompt_ids']).encode()).hexdigest(),
        'ttft_seconds': stats([row['ttft_s'] for row in measured]),
        'decode_tok_s': stats([1 / row['mean_tpot_s'] for row in measured]),
        'input_over_ttft_tok_s': stats([config['prompt_tokens'] / row['ttft_s'] for row in measured]),
        'unique_output_texts': len(set(row['text'] for row in measured)),
        'output_sha256': [hashlib.sha256(row['text'].encode()).hexdigest() for row in measured],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eager', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-name', default='eager')
    parser.add_argument('--candidate-name', default='graph')
    args = parser.parse_args()
    if args.baseline_name == args.candidate_name:
        parser.error('Baseline and candidate names must differ')
    eager, graph = summarize(args.eager), summarize(args.graph)
    assert eager['prompt_ids_sha256'] == graph['prompt_ids_sha256']
    assert eager['output_tokens'] == graph['output_tokens']
    result = {
        'scope': 'Sequential HTTP measurements; warmup excluded. Input/TTFT is not pure kernel prefill. '
                 'Text hashes are observational, not a model accuracy gate.',
        args.baseline_name: eager, args.candidate_name: graph,
        'median_decode_ratio': graph['decode_tok_s']['median'] / eager['decode_tok_s']['median'],
        'median_ttft_ratio': graph['ttft_seconds']['median'] / eager['ttft_seconds']['median'],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
