"""从原始JSONL生成同输入配对表；clean C16未完成时拒绝给结论。"""
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent


def requests(kv, length):
    rows = [json.loads(s) for s in (OUT / f'{kv}-p{length}-o128.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r.get('kind') == 'request']
    assert len(rows) == (1 if length == 112000 else 3)
    for r in rows:
        assert r['valid'] and r['response']['usage']['prompt_tokens'] == length
        assert r['response']['usage']['completion_tokens'] == 128
        assert r['prefix_cache_delta']['vllm:prefix_cache_hits_total'] == 0
    return rows


def main():
    summary = []
    for n in (8192, 32768, 112000):
        a, b = requests('c8', n), requests('c16-clean', n)
        assert [r['prompt_ids_sha256'] for r in a] == [r['prompt_ids_sha256'] for r in b]
        row = {'prompt_tokens': n, 'samples_per_kv': len(a), 'output_tokens': 128,
               'c16_source': 'clean rerun after stopping old systemd retry loop',
               'outputs_identical_pairs': sum(x['response']['text'] == y['response']['text'] for x, y in zip(a,b))}
        for name, rows in (('c8', a), ('c16', b)):
            row[name] = {key + '_median': statistics.median(r[key] for r in rows)
                         for key in ('prefill_tok_s','decode_tok_s')}
            row[name]['mtp_acceptance'] = [r['spec_decode']['acceptance_rate'] for r in rows]
        row['c8_decode_loss_percent_vs_c16'] = 100 * (1 - row['c8']['decode_tok_s_median'] / row['c16']['decode_tok_s_median'])
        row['c8_prefill_change_percent_vs_c16'] = 100 * (row['c8']['prefill_tok_s_median'] / row['c16']['prefill_tok_s_median'] - 1)
        summary.append(row)
    (OUT / 'paired-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
