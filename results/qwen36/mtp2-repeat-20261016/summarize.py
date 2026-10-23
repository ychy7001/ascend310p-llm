"""核对重复冷请求，比较逐位置接受数与上轮固定输入/输出。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'mtp2-history-20261016' / 'draft-eager'
summary = {'scope': 'six cold 32256/128 requests in one fresh draft-eager process; not latency percentiles',
           'requests': []}
first = {}
for cycle in range(3):
    for index in range(2):
        path = ROOT / f'cycle{cycle}-input{index}.json'
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        assert d['response']['usage']['prompt_tokens'] == 32256
        assert d['response']['usage']['completion_tokens'] == 128
        assert d['cache_after']['vllm:prefix_cache_hits_total'] == d['cache_before']['vllm:prefix_cache_hits_total']
        delta = d['metric_delta']
        drafts = delta['vllm:spec_decode_num_drafts_total']
        accepted = [delta[f'accepted_pos_{i}'] for i in range(2)]
        assert sum(accepted) == delta['vllm:spec_decode_num_accepted_tokens_total']
        if cycle == 0:
            first[index] = d
        a = first[index]
        assert a['prompt_ids_sha256'] == d['prompt_ids_sha256']
        oldpath = OLD / f'{"before" if cycle == 0 else "after"}-{index}.json'
        old = json.loads(oldpath.read_text())
        assert old['prompt_ids_sha256'] == d['prompt_ids_sha256']
        summary['requests'].append({
            'cycle': cycle, 'input_index': index,
            'prefill_tok_s': d['prefill_tok_s'], 'decode_tok_s': d['decode_tok_s'],
            'drafts': drafts, 'accepted_per_pos': accepted,
            'acceptance_per_pos': [n / drafts if drafts else None for n in accepted],
            'identical_output_to_cycle0': a['response']['text'] == d['response']['text'],
            'identical_output_to_previous_run': old['response']['text'] == d['response']['text'],
        })
print(json.dumps(summary, indent=2, ensure_ascii=False))
