# SPDX-License-Identifier: Apache-2.0
"""Basic real-checkpoint HTTP generation gate, not a model quality benchmark."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:18038')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    results=[]
    tests=[('请只输出数字：17 加 25 等于多少？','42'),('请用中文用一句话解释计算机内存的作用。',None),('Return only the word READY.', 'READY')]
    for prompt, expected in tests:
        body={'model':'qwen38-w4a8-tp2-text','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}}
        req=urllib.request.Request(args.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=600) as response:
            data=json.load(response)
        elapsed=time.perf_counter()-start
        message=data['choices'][0]['message']
        content=message.get('content') or ''
        result={'prompt':prompt,'expected_substring':expected,'elapsed_seconds':elapsed,'response':data,'basic_pass':bool(content.strip()) and (expected is None or expected in content)}
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        args.output.write_text(json.dumps({'scope':'basic smoke only; not an accuracy or decode benchmark','results':results},ensure_ascii=False,indent=2))
    if not all(r['basic_pass'] for r in results):
        raise AssertionError('Basic generation gate failed')

if __name__=='__main__': main()
