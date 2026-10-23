# SPDX-License-Identifier: Apache-2.0
"""从仓库根目录执行，复核相同输入及单流phase counter中位数。"""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
baseline = root.parent / 'pr17640-b4096-20261013'
result = []
for tokens, reference in [(8192, 'recheck-p8192-o128.jsonl'), (32256, 'p32256-o128.jsonl')]:
    def read(path):
        return [r for line in path.read_text().splitlines()
                if (r := json.loads(line))['kind'] == 'request']
    before = read(baseline / reference)
    after = read(root / f'mindieq-native-layout-phases-p{tokens}-o128.jsonl')
    assert len(before) == len(after) == 3
    assert all(r['valid'] for r in before + after)
    assert [r['prompt_ids_sha256'] for r in before] == [r['prompt_ids_sha256'] for r in after]
    def stats(rows):
        return {k: median(r[k] for r in rows) for k in ['prefill_tok_s', 'decode_tok_s']}
    a, b = stats(before), stats(after)
    result.append({'prompt_tokens': tokens, 'output_tokens': 128, 'repeats': 3,
        'baseline': a, 'mindieq_native_layout': b,
        'relative_change': {k: b[k] / a[k] - 1 for k in a},
        'identical_prompt_hashes': True,
        'identical_texts': sum(a['response']['text'] == b['response']['text'] for a,b in zip(before,after)),
        'scope': '同日旧结果对照；尚非交替A/B；服务phase counter，单流、prefix off、warmup1'})
(root / 'service-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
