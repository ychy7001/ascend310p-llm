# SPDX-License-Identifier: Apache-2.0
"""D4已完成业务和8K阶段分析；对照C3及历史native，不作为全面质量评分。"""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

sources = {
    'native_historical': root.parent / 'pr17640-mtp-20261013' / 'mtp2-chat.jsonl',
    'C3_min17': root.parent / 'mindieq-a8-service-20261014' / 'chat.jsonl',
    'D4_min128': root / 'chat.jsonl',
}
chats = {name: read(path) for name, path in sources.items()}
for key in ('repeats', 'warmup', 'max_tokens', 'temperature', 'enable_thinking'):
    assert len({rows[0][key] for rows in chats.values()}) == 1, key
result = {'scope': 'D4与C3连续独立启动及历史native对照；非交替A/B，非完整质量评分', 'chat': [], 'full_gsm64': 'pending; not included in this analysis'}
for case in ('dialogue', 'meeting', 'code', 'analysis'):
    data = {name: [row for row in rows if row['kind'] == 'request' and row['case'] == case] for name, rows in chats.items()}
    assert all(len(rows) == 3 and all(row['valid'] for row in rows) for rows in data.values())
    assert all([row['prompt_sha256'] for row in rows] == [row['prompt_sha256'] for row in data['D4_min128']] for rows in data.values())
    stats = {name: {key: median(row[key] for row in rows) for key in ('ttft_s', 'decode_tok_s', 'elapsed_s', 'acceptance_rate')} for name, rows in data.items()}
    result['chat'].append({'case': case, 'stats': stats, 'finish_reason': {name: [row['finish_reason'] for row in rows] for name, rows in data.items()}, 'completion_tokens': {name: [row['usage']['completion_tokens'] for row in rows] for name, rows in data.items()}, 'identical_to_native': sum(a['text'] == b['text'] for a, b in zip(data['D4_min128'], data['native_historical']))})
phase_sources = {'A1_native': root.parent / 'mindieq-paired-20261014' / 'A1-p8192-o128.jsonl', 'C3_min17': root.parent / 'mindieq-a8-service-20261014' / 'p8192-o128.jsonl', 'D4_min128': root / 'p8192-o128.jsonl'}
phases = {name: [row for row in read(path) if row['kind'] == 'request'] for name, path in phase_sources.items()}
assert all(len(rows) == 3 and all(row['valid'] for row in rows) for rows in phases.values())
assert all([row['prompt_ids_sha256'] for row in rows] == [row['prompt_ids_sha256'] for row in phases['D4_min128']] for rows in phases.values())
result['p8192'] = {name: {key: median(row[key] for row in rows) for key in ('prefill_tok_s', 'decode_tok_s')} for name, rows in phases.items()}
for name, rows in phases.items():
    accepted = sum(row['spec_decode']['accepted_tokens'] for row in rows)
    drafted = sum(row['spec_decode']['drafted_tokens'] for row in rows)
    result['p8192'][name]['acceptance_rate'] = accepted / drafted
boundary = read(root / 'quality-boundary.jsonl')
assert boundary[-1]['kind'] == 'summary'
answers = [row for row in boundary if row['kind'] == 'answer']
assert {row['index'] for row in answers} == {711, 205}
result['quality_boundary'] = [{'index': row['index'], 'gold': row['gold'], 'predicted': row['predicted'], 'correct': row['correct'], 'finish_reason': row['finish_reason']} for row in answers]
result['sanity'] = json.loads((root / 'sanity.json').read_text())['status']
assert result['sanity'] == 'PASS'
(root / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
