"""验证同输入、零prefix命中及逐位置接受数，汇总实际完成的请求。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
summary = {'scope': '32K/128 cold requests; fixed 112K service; two inputs per phase', 'modes': {}}
paired = {}
for mode in ('fullgraph', 'draft-eager'):
    rows = []
    for phase in ('before', 'after'):
        for index in range(2):
            path = ROOT / mode / f'{phase}-{index}.json'
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            assert d['response']['usage']['completion_tokens'] == 128
            assert d['cache_after']['vllm:prefix_cache_hits_total'] == d['cache_before']['vllm:prefix_cache_hits_total']
            delta = d['metric_delta']
            drafts = delta['vllm:spec_decode_num_drafts_total']
            accepted = [delta[f'accepted_pos_{i}'] for i in range(2)]
            assert sum(accepted) == delta['vllm:spec_decode_num_accepted_tokens_total']
            rows.append({'phase': phase, 'index': index,
                         'prefill_tok_s': d['prefill_tok_s'], 'decode_tok_s': d['decode_tok_s'],
                         'drafts': drafts, 'accepted_per_pos': accepted,
                         'acceptance_per_pos': [n / drafts if drafts else None for n in accepted]})
            paired[(mode, phase, index)] = d
    summary['modes'][mode] = rows
summary['comparisons'] = []
summary['history_comparisons'] = []
for mode in ('fullgraph', 'draft-eager'):
    for index in range(2):
        a = paired.get((mode, 'before', index))
        b = paired.get((mode, 'after', index))
        if a is not None and b is not None:
            assert a['prompt_ids_sha256'] == b['prompt_ids_sha256']
            summary['history_comparisons'].append({'mode': mode, 'index': index,
                                                  'identical_output': a['response']['text'] == b['response']['text']})
for phase in ('before', 'after'):
    for index in range(2):
        a = paired.get(('fullgraph', phase, index))
        b = paired.get(('draft-eager', phase, index))
        if a is not None and b is not None:
            assert a['prompt_ids_sha256'] == b['prompt_ids_sha256']
            summary['comparisons'].append({'phase': phase, 'index': index,
                                          'identical_input': True,
                                          'identical_output': a['response']['text'] == b['response']['text']})
print(json.dumps(summary, indent=2, ensure_ascii=False))
