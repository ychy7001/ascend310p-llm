# SPDX-License-Identifier: Apache-2.0
"""Compare exclusive-endpoint phase measurements with matching prompt protocols.

This verifies the recorded warmup prompt, protocol, token usage and per-request
prompt hashes when available. Legacy files omit the measured prompt hashes;
the report explicitly identifies that limitation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    configs = [r for r in rows if r['kind'] == 'config']
    requests = [r for r in rows if r['kind'] == 'request']
    if len(configs) != 1:
        raise ValueError(f'{path}: expected one config')
    config = configs[0]
    if len(requests) != config['repeats']:
        raise ValueError(f'{path}: incomplete measured requests')
    for row in requests:
        usage = row['response']['usage']
        if not (row['valid'] and usage['prompt_tokens'] == config['prompt_tokens']
                and usage['completion_tokens'] == config['output_tokens']):
            raise ValueError(f'{path}: invalid phase counter or token usage')
        if any(row[key] is None or row[key] <= 0
               for key in ('prefill_tok_s', 'decode_tok_s')):
            raise ValueError(f'{path}: missing positive throughput')
    return config, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base_config, base_rows = read(args.baseline)
    candidate_config, candidate_rows = read(args.candidate)
    fields = ('model', 'prompt_tokens', 'output_tokens', 'repeats',
              'protocol', 'unique_prompts', 'prompt_ids')
    mismatches = [key for key in fields if base_config.get(key) != candidate_config.get(key)]
    if mismatches:
        raise ValueError(f'Unmatched measurement protocol: {mismatches}')
    base_hashes = [r.get('prompt_ids_sha256') for r in base_rows]
    candidate_hashes = [r.get('prompt_ids_sha256') for r in candidate_rows]
    hashes_available = all(base_hashes) and all(candidate_hashes)
    if hashes_available and base_hashes != candidate_hashes:
        raise ValueError('Measured prompt hashes differ')
    result = {
        'baseline': str(args.baseline), 'candidate': str(args.candidate),
        'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        'prompt_tokens': base_config['prompt_tokens'],
        'output_tokens': base_config['output_tokens'],
        'repeats': base_config['repeats'],
        'recorded_warmup_prompt_equal': True,
        'measured_prompt_hashes_equal': True if hashes_available else None,
        'limits': 'Sequential pair; no reliable P95 or business accuracy claim. '
                  + ('' if hashes_available else 'Measured prompt hashes missing from legacy evidence. ')
                  +
                  'Service settings and artifacts must be checked against startup logs.',
    }
    for phase in ('prefill', 'decode'):
        key = phase + '_tok_s'
        base = [r[key] for r in base_rows]
        candidate = [r[key] for r in candidate_rows]
        result[phase] = {
            'baseline_samples_tok_s': base,
            'candidate_samples_tok_s': candidate,
            'baseline_median_tok_s': median(base),
            'candidate_median_tok_s': median(candidate),
            'median_change_percent': 100 * (median(candidate) / median(base) - 1),
        }
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
