# SPDX-License-Identifier: Apache-2.0
"""Small quality/reuse probe; store raw responses, never claim a full accuracy eval."""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

CASES = [
    ('multiply', 'Compute 37 * 19. Output only the integer answer.', '703'),
    ('add', 'Compute 128 + 256. Output only the integer answer.', '384'),
    ('subtract', 'Compute 1000 - 237. Output only the integer answer.', '763'),
    ('divide', 'Compute 144 / 12. Output only the integer answer.', '12'),
    ('sort', 'Sort these integers ascending: 9, 2, 7. Output only 2,7,9 without spaces.', '2,7,9'),
    ('extract', 'The access code is NPU7329. Output only the access code.', 'NPU7329'),
    ('中文', '计算 17 加 26。只输出阿拉伯数字答案。', '43'),
    ('boolean', 'Is 17 a prime number? Output only YES or NO.', 'YES'),
]


def post(base, path, body):
    request = urllib.request.Request(base.rstrip('/')+path,
        data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def answer(base, model, label, prompt, expected):
    start = time.perf_counter()
    response = post(base, '/v1/chat/completions', {
        'model':model, 'messages':[{'role':'user','content':prompt}],
        'temperature':0, 'max_tokens':128,
        'chat_template_kwargs':{'enable_thinking':False},
    })
    choice = response['choices'][0]
    text = choice['message'].get('content') or ''
    # Strict answer adherence is reported separately from answer presence.
    compact = re.sub(r'\s+', '', text).strip('`*.。')
    return {'label':label, 'prompt':prompt, 'expected':expected, 'text':text,
        'strict_match':compact==expected, 'answer_present':bool(re.search(
            r'(?<![A-Za-z0-9])'+re.escape(expected)+r'(?![A-Za-z0-9])', text)),
        'usage':response.get('usage'), 'finish_reason':choice.get('finish_reason'),
        'elapsed_s':time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    with args.out.open('x') as stream:
        rows=[answer(args.base_url,args.model,*case) for case in CASES]
        before=answer(args.base_url,args.model,*CASES[5])
        tokens=post(args.base_url,'/tokenize',{'model':args.model,
            'prompt':'Unrelated background: the replacement code is ZXQ9182. '*4096,
            'add_special_tokens':False})['tokens'][:3072]
        interference=post(args.base_url,'/v1/completions',{'model':args.model,
            'prompt':tokens, 'max_tokens':64, 'temperature':0,'ignore_eos':True})
        after=answer(args.base_url,args.model,*CASES[5])
        result={'scope':'8 toy questions plus A/B/A reuse probe, not formal accuracy or allocator isolation proof',
            'model':args.model,'base_url':args.base_url,'rows':rows,
            'strict_matches':sum(r['strict_match'] for r in rows),
            'answers_present':sum(r['answer_present'] for r in rows),
            'reuse':{'before':before,'after':after,
                'identical_text':before['text']==after['text'],
                'canary_leaked':'ZXQ9182' in after['text'],
                'interference_usage':interference.get('usage')},
        }
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ['strict_matches','answers_present']}))
    print(json.dumps({k:result['reuse'][k] for k in ['identical_text','canary_leaked']}))


if __name__=='__main__':
    main()
