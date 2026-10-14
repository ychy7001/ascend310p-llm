"""相同输入SHA/零prefix hit的RC与EP模型对照；不宣称纯模式因果。"""
import json
from pathlib import Path
from statistics import median

base = Path(__file__).parent/'rc-ep-pair'
for tokens in (2304,8192,32000):
    runs = {}
    for platform in ('rc','ep'):
        rows = [json.loads(s) for s in (base/platform/f'p{tokens}.jsonl').read_text().splitlines()]
        runs[platform] = [r for r in rows if r['kind']=='request']
        assert len(runs[platform]) == 3
        for r in runs[platform]:
            assert r['valid'] and r['prefix_cache_delta']['vllm:prefix_cache_hits_total']==0
            assert r['response']['usage']['prompt_tokens']==tokens
            assert r['response']['usage']['completion_tokens']==128
    assert all(a['prompt_ids_sha256']==b['prompt_ids_sha256']
               for a,b in zip(runs['rc'],runs['ep']))
    print(json.dumps({'prompt_tokens':tokens,'output_tokens':128,'repeats':3,
        'median':{p:{k:median(r[k] for r in rows) for k in ('prefill_tok_s','decode_tok_s')}
                  for p,rows in runs.items()},
        'identical_outputs':sum(a['response']['text']==b['response']['text']
                  for a,b in zip(runs['rc'],runs['ep']))},ensure_ascii=False))
