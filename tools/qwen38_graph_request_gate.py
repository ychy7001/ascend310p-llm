# SPDX-License-Identifier: Apache-2.0
"""Closed-answer sequential requests to exercise graph input/state refresh; not an accuracy suite."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18038')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tests = [
        ('Return only the result of 19 + 23.', '42'),
        ('Return only the result of 15 * 7.', '105'),
    ]
    for code, repeats in [('A7K9', 0), ('B3M8', 500), ('C4N2', 10), ('D8P5', 3000),
                          ('E6R1', 0), ('F2T7', 100)]:
        filler = ' filler' * repeats
        tests.append((f'The secret code is {code}.\n{filler}\n'
                      'Return only the secret code from the first sentence.', code))
    results = []
    for prompt, expected in tests:
        payload = {'model': 'qwen38-w4a8-tp2-text',
                   'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': 0, 'max_tokens': 32,
                   'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(args.url + '/v1/chat/completions',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
        content = data['choices'][0]['message'].get('content') or ''
        result = {'prompt': prompt, 'expected': expected, 'content': content,
                  'elapsed_seconds': time.perf_counter() - start, 'response': data,
                  'pass': content.strip() == expected}
        results.append(result)
        args.output.write_text(json.dumps({'scope': __doc__, 'results': results},
                                          ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ('prompt', 'response')},
                         ensure_ascii=False), flush=True)
    assert all(result['pass'] for result in results), 'Sequential request refresh gate failed'


if __name__ == '__main__':
    main()
