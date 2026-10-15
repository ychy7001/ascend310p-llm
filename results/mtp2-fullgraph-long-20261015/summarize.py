"""核对成对输入/零缓存命中并汇总实际完成的测量。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path):
    return [d for line in path.read_text().splitlines()
            if (d := json.loads(line)).get('kind') == 'request']


summary = {'scope': '2 measured requests per length; not latency percentiles or formal accuracy', 'lengths': {}}
for count in (32256, 114176):
    pair = {}
    for mtp in (1, 2):
        path = ROOT / f'mtp{mtp}' / f'bench-{count}.jsonl'
        if path.exists():
            data = rows(path)
            for d in data:
                assert d['valid']
                assert d['prefix_cache_delta']['vllm:prefix_cache_hits_total'] == 0
            pair[f'mtp{mtp}'] = data
    if all(len(pair.get(f'mtp{m}', [])) == 2 for m in (1, 2)):
        a, b = pair['mtp1'], pair['mtp2']
        assert [d['prompt_ids_sha256'] for d in a] == [d['prompt_ids_sha256'] for d in b]
        summary['lengths'][str(count)] = {
            'matched_prompt_hashes': True,
            'identical_outputs': [x['response']['text'] == y['response']['text'] for x, y in zip(a, b)],
            **{key: [{k: d[k] for k in ('repeat', 'prefill_tok_s', 'decode_tok_s', 'spec_decode')}
                     for d in data] for key, data in pair.items()},
        }
canaries = {}
for label, folder in (('mtp1', 'mtp1'), ('mtp2', 'mtp2-recheck')):
    path = ROOT / folder / 'meeting-114176.json'
    if path.exists():
        d = json.loads(path.read_text())
        assert d['score']['passed']
        canaries[label] = d
if len(canaries) == 2:
    a, b = canaries['mtp1'], canaries['mtp2']
    assert a['prompt_ids_sha256'] == b['prompt_ids_sha256']
    summary['meeting_114176'] = {
        'scope': 'single short-answer canaries; request histories differ; not fixed-output throughput',
        'identical_output': a['result']['choices'][0]['text'] == b['result']['choices'][0]['text'],
        **{label: {k: d[k] for k in ('prefill_tok_s', 'decode_tok_s', 'elapsed_s', 'score')}
           for label, d in canaries.items()},
    }
print(json.dumps(summary, indent=2, ensure_ascii=False))
