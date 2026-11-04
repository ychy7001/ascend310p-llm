#!/usr/bin/env python3
"""冻结的标点/排序诊断；不替代短题、业务或性能验收。"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
cases = [
    ('sort-original', 'Sort these integers ascending: 9, 2, 7. Output only 2,7,9 without spaces.', '2,7,9'),
    ('copy-commas', 'Output exactly this text and nothing else: 2,7,9', '2,7,9'),
    ('copy-letters', 'Output exactly this text and nothing else: a,b,c', 'a,b,c'),
    ('copy-semicolons', 'Output exactly this text and nothing else: 2;7;9', '2;7;9'),
    ('copy-array', 'Output exactly this text and nothing else: [2,7,9]', '[2,7,9]'),
    ('sort-json', 'Sort these integers ascending: 9, 2, 7. Return only a JSON array of three integers.', [2,7,9]),
]
with a.out.open('x') as stream:
    for case, prompt, expected in cases:
        for repeat in range(3):
            payload = {'model':a.model, 'messages':[{'role':'user','content':prompt}],
                       'temperature':0, 'seed':0, 'max_tokens':128,
                       'cache_salt':f'dense-format-{case}-r{repeat}',
                       'chat_template_kwargs':{'enable_thinking':False},
                       'logprobs':True, 'top_logprobs':5, 'return_token_ids':True}
            if case == 'sort-json':
                payload['response_format'] = {'type':'json_schema','json_schema':{
                    'name':'sorted_integers','strict':True,'schema':{
                        'type':'array','items':{'type':'integer'},'minItems':3,'maxItems':3}}}
            row = {'case':case, 'repeat':repeat, 'request':payload, 'expected':expected,
                   'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
            try:
                req=urllib.request.Request(a.base_url+'/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=120) as response: row['response']=json.load(response)
                choice=row['response']['choices'][0]
                text=choice['message'].get('content') or ''
                row['text']=text
                actual=json.loads(text) if isinstance(expected,list) else text.strip()
                row['matches_expected']=actual==expected and choice['finish_reason']=='stop'
            except Exception as error:
                row['error']=str(error)
                if isinstance(error,urllib.error.HTTPError):row['http_error_body']=error.read().decode()
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            print(json.dumps({k:v for k,v in row.items() if k not in ('request','response')},ensure_ascii=False),flush=True)
