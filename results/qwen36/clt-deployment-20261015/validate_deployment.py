"""在目标机独占端点执行阶段测速、112K检索和短请求回归；保留原始数据。"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--deployment', type=Path, required=True)
a = p.parse_args()
root = a.deployment
logs = root / 'logs'
sys.path.insert(0, str(root / 'repo/tools'))
from bench_http_stream import post, request
from bench_server_phases import snapshot, METRICS, SPEC_METRICS

BASE = 'http://127.0.0.1:18016'
MODEL = 'qwen36-w4a8-c8'


def emit(row):
    with (logs / 'validation-progress.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def memory(stage):
    r = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True)
    (logs / f'memory-{stage}.txt').write_text(r.stdout + r.stderr)


def tokenize(text):
    with post(BASE, '/tokenize', {'model': MODEL, 'prompt': text,
                                'add_special_tokens': False}) as r:
        return json.load(r)['tokens']


memory('before')
for n in (8192, 32768):
    emit({'stage': 'benchmark_start', 'prompt_tokens': n})
    subprocess.run([
        'python3', str(root / 'repo/tools/bench_server_phases.py'),
        '--base-url', BASE, '--model', MODEL,
        '--out', str(logs / f'112k-bench-p{n}-v2.jsonl'),
        '--prompt-tokens', str(n), '--output-tokens', '128', '--repeats', '3',
        '--unique-prompts', '--spec-metrics',
    ], check=True)
    memory(str(n))
    emit({'stage': 'benchmark_complete', 'prompt_tokens': n})

# Exact token length. Each case uses a distinct canary and no prefix cache.
# Fixed 128-token output is for timing; check the leading answer separately.
filler = tokenize('Routine meeting note. The team discussed scheduling and delivery. ' * 15000)
prefix = tokenize('<|im_start|>user\nRead the following meeting archive. Find the '
                  'ARCHIVE_ACCESS_CODE and reply with only its value.\n<archive>\n')
suffix = tokenize('\n</archive>\nWhat is ARCHIVE_ACCESS_CODE? Reply with only its '
                  'value and no explanation.<|im_end|>\n<|im_start|>assistant\n'
                  '<think>\n\n</think>\n\n')
passed = True
for label, fraction, expected in [('early', .1, 'ASCEND732941'),
                                  ('late', .9, 'MEETING586203')]:
    needle = tokenize(f'\nARCHIVE_ACCESS_CODE = {expected}\n')
    count = 112000 - len(prefix) - len(suffix) - len(needle)
    offset = int(count * fraction)
    tokens = prefix + filler[:offset] + needle + filler[offset:count] + suffix
    assert len(tokens) == 112000
    emit({'stage': 'needle_start', 'label': label, 'prompt_tokens': len(tokens)})
    before = snapshot(BASE, True)
    result = request(BASE, MODEL, tokens, 128)
    after = snapshot(BASE, True)
    delta = {key: after[key] - before[key] for key in before}
    prefill, decode = delta[METRICS[1]], delta[METRICS[2]]
    valid = delta[METRICS[0]] == 1 and prefill > 0 and decode > 0
    answer_ok = result['text'].lstrip().startswith(expected)
    row = {'label': label, 'prompt_tokens': len(tokens), 'output_tokens': 128,
           'needle_fraction': fraction, 'expected': expected,
           'prompt_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
           'protocol': 'warm service; no same-length warmup; prefix disabled; one sample per distinct needle',
           'valid_metrics': valid, 'answer_starts_with_expected': answer_ok,
           'metrics_before': before, 'metrics_after': after,
           'prefill_tok_s': len(tokens) / prefill if valid else None,
           'decode_tok_s': 127 / decode if valid else None,
           'spec_decode': {key: delta[key] for key in SPEC_METRICS},
           'response': result}
    (logs / f'112k-needle-{label}.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
    emit({k: row[k] for k in ('label', 'prefill_tok_s', 'decode_tok_s',
                              'valid_metrics', 'answer_starts_with_expected')})
    memory(f'needle-{label}')
    passed = passed and valid and answer_ok

# Revisit A after B to detect obvious recurrent-state contamination.
answers = []
for expression, expected in [('37 * 19', '703'), ('1000 - 237', '763'),
                             ('37 * 19', '703'), ('18 * 7', '126'),
                             ('384 + 256', '640'), ('81 / 9', '9'),
                             ('17 + 26', '43'), ('12 * 12', '144'), ('2 + 2', '4')]:
    with post(BASE, '/v1/chat/completions', {
        'model': MODEL, 'messages': [{'role': 'user', 'content':
            f'Calculate {expression}. Reply with only the number.'}],
        'temperature': 0, 'max_tokens': 16,
        'chat_template_kwargs': {'enable_thinking': False},
    }) as r:
        payload = json.load(r)
    actual = payload['choices'][0]['message']['content'].strip()
    answers.append({'expression': expression, 'expected': expected,
                    'actual': actual, 'pass': actual == expected})
(logs / '112k-post-long-fixed-answers.json').write_text(json.dumps(answers, indent=2))
passed = passed and all(x['pass'] for x in answers)
emit({'stage': 'complete', 'pass': passed,
      'post_long_fixed_matches': sum(x['pass'] for x in answers),
      'post_long_fixed_cases': len(answers)})
raise SystemExit(0 if passed else 1)
