"""从保留的原始结果生成配对统计；不执行 NPU 请求。"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'clt-deployment-20261015'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def bench(path):
    samples = [r for r in rows(path) if r.get('kind') == 'request']
    assert len(samples) == 3 and all(r['valid'] for r in samples)
    assert all(r['response']['usage']['completion_tokens'] == 128 for r in samples)
    return {
        'samples': len(samples),
        **{k: statistics.median(r[k] for r in samples)
           for k in ('prefill_tok_s', 'decode_tok_s')},
        'prompt_hashes': [r['prompt_ids_sha256'] for r in samples],
        'acceptance_rates': [r['spec_decode']['acceptance_rate'] for r in samples],
    }


result = {'protocol': 'TP1, W4A8, MTP1, FULL_DECODE_ONLY Graph, max_model_len=114688, util=.90, prefix off; 128 output tokens', 'short': [], 'long': []}
for n in (8192, 32768):
    a = bench(BASE / f'112k-bench-p{n}-v2.jsonl')
    b = bench(ROOT / f'c16-p{n}.jsonl')
    assert a['prompt_hashes'] == b['prompt_hashes']
    result['short'].append({'prompt_tokens': n, 'c8': a, 'c16': b,
                           'c8_decode_loss_percent': 100 * (1-a['decode_tok_s']/b['decode_tok_s']),
                           'c8_prefill_loss_percent': 100 * (1-a['prefill_tok_s']/b['prefill_tok_s'])})
for label in ('early', 'late'):
    a = json.loads((BASE / f'112k-needle-{label}.json').read_text())
    b = json.loads((ROOT / f'c16-needle-{label}.json').read_text())
    assert a['prompt_sha256'] == b['prompt_sha256']
    assert a['valid_metrics'] and b['valid_metrics']
    entry = {'label': label, 'prompt_tokens': 112000}
    for name, data in (('c8', a), ('c16', b)):
        assert data['response']['usage']['completion_tokens'] == 128
        spec = data['spec_decode']
        entry[name] = {k: data[k] for k in ('prefill_tok_s', 'decode_tok_s', 'answer_starts_with_expected')}
        entry[name]['acceptance_rate'] = spec['vllm:spec_decode_num_accepted_tokens_total']/spec['vllm:spec_decode_num_draft_tokens_total']
    entry['c8_decode_loss_percent'] = 100*(1-a['decode_tok_s']/b['decode_tok_s'])
    entry['c8_prefill_loss_percent'] = 100*(1-a['prefill_tok_s']/b['prefill_tok_s'])
    result['long'].append(entry)
result['c8_restored_8192'] = bench(ROOT / 'c8-restored-8192.jsonl')
(ROOT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
