# SPDX-License-Identifier: Apache-2.0
"""诊断：共享文档A/B/A后缀复用与独立缓存salt冷参考；非完整长文质量评估。"""
import argparse
import json
import uuid
from pathlib import Path
from bench_prefix_reuse import post, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prefix-tokens', type=int, default=8192)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    def tokenize(text):
        return post(args.base_url, '/tokenize', {
            'model': args.model, 'prompt': text,
            'add_special_tokens': False,
        })['tokens']

    start = tokenize('Prefix-state diagnostic v1. Read the document and answer the final question.\nDocument:\n')
    marker = tokenize('\nThe secret code is ASCEND732941. The project owner is LinMei.\n')
    budget = args.prefix_tokens - len(start) - len(marker)
    if budget < 1:
        raise ValueError('Prefix too short')
    filler = tokenize('Ordinary note: calibration and memory measurements are recorded carefully. ' * budget)[:budget]
    common = start + filler[:budget//2] + marker + filler[budget//2:]
    assert len(common) == args.prefix_tokens
    tails = {
        'A': ('\nEnd of document. What is the secret code? Reply with only the code.\nAnswer:', 'ASCEND732941'),
        'B': ('\nEnd of document. Who is the project owner? Reply with only the name.\nAnswer:', 'LinMei'),
    }
    tokens = {key: common + tokenize(tail) for key, (tail, _) in tails.items()}
    rows = []
    shared_salt = 'prefix-state-' + uuid.uuid4().hex
    cold_salt = 'prefix-cold-' + uuid.uuid4().hex
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps({'kind': 'config', 'prefix_tokens': len(common),
                                 'prompts': tokens, 'max_tokens': {'A': 32, 'B': 256},
                                 'protocol': __doc__}) + '\n')
        for label, key, _ in [('A-cold', 'A', True), ('A-hot', 'A', False),
                                  ('B-shared', 'B', False), ('A-after-B', 'A', False),
                                  ('B-cold', 'B', True)]:
            salt = cold_salt if label == 'B-cold' else shared_salt
            row = request(args.base_url, args.model, tokens[key], 32 if key == 'A' else 256, salt)
            answer = row['text'].rsplit('</think>', 1)[-1].strip()
            row.update(kind='request', label=label, expected=tails[key][1],
                       final_answer=answer, correct=answer == tails[key][1], cache_salt=salt)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({k: row[k] for k in ('label', 'correct', 'text', 'wall_seconds')}, ensure_ascii=False), flush=True)
        summary = {'kind': 'summary', 'all_valid': all(r['valid'] for r in rows),
                   'all_correct': all(r['correct'] for r in rows),
                   'A_texts_equal': len({r['text'] for r in rows if r['label'].startswith('A')}) == 1,
                   'B_shared_equals_cold': rows[2]['text'] == rows[4]['text']}
        stream.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
