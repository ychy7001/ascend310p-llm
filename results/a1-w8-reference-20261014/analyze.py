# SPDX-License-Identifier: Apache-2.0
"""正式cold结果必须全量、无缓存命中、与E5输入SHA相同。"""
import json
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parent
result=[]
for tokens in (8192,32256):
    rows=[json.loads(line) for line in (root/f'cold-p{tokens}-o128.jsonl').read_text().splitlines()]
    data=[row for row in rows if row['kind']=='request']
    other=[json.loads(line) for line in (root.parent/'mindieq-shared-gate-20261014'/'service'/f'p{tokens}-o128.jsonl').read_text().splitlines() if json.loads(line)['kind']=='request']
    assert len(data)==len(other)==3
    assert all(row['valid'] and row['prefix_cache_delta']['vllm:prefix_cache_hits_total']==0 and row['prefix_cache_delta']['vllm:prefix_cache_queries_total']==tokens for row in data)
    assert [row['prompt_ids_sha256'] for row in data]==[row['prompt_ids_sha256'] for row in other]
    result.append({'tokens':tokens,'a1_w8_tp2':{key:median(row[key] for row in data) for key in ('prefill_tok_s','decode_tok_s')},'a2_w4_tp1_c8_mtp2':{key:median(row[key] for row in other) for key in ('prefill_tok_s','decode_tok_s')},'cache_hits_each':[row['prefix_cache_delta']['vllm:prefix_cache_hits_total'] for row in data]})
(root/'comparison.json').write_text(json.dumps({'scope':'跨机器/TP/软件/KV/MTP配置的部署参考，不是纯W4vsW8或同DUO验收','results':result},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
