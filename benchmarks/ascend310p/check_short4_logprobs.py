#!/usr/bin/env python3
"""冻结short-4 C1/C3各3轮；相同批次组成，保留raw logprobs及首分歧。"""
import argparse,concurrent.futures,copy,hashlib,json,threading,time,urllib.request
from pathlib import Path
from check_action_state_regression import SCHEMA,canonical

def post(base,path,payload):
    req=urllib.request.Request(base+path,json.dumps(payload).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as r:return json.load(r)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',required=True);p.add_argument('--model',required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    source=json.loads(a.fixtures.read_text());cases={c['id']:c for c in source['cases']};plan=[]
    for repeat in range(3):
        for concurrency in (1,3):
            ids=['short-4'] if concurrency==1 else ['short-3','short-4','short-5']
            batch=[]
            for case_id in ids:
                case=cases[case_id]
                payload={'model':a.model,'messages':[{'role':'user','content':case['prompt']}],'temperature':0,'seed':0,'max_tokens':512,'cache_salt':f'short4-logprobs-v1-r{repeat}-c{concurrency}-{case_id}','chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'action_state','strict':True,'schema':SCHEMA}},'logprobs':True,'top_logprobs':20,'return_token_ids':True}
                batch.append({'repeat':repeat,'concurrency':concurrency,'id':case_id,'request':payload,'expected':case['expected']})
            plan.append(batch)
    (a.out/'frozen-plan.json').write_text(json.dumps({'fixture_sha256':hashlib.sha256(a.fixtures.read_bytes()).hexdigest(),'batches':plan,'scope':'Finite diagnostic; client concurrency does not imply actual kernel batch; scheduler trace required. Raw logprobs, not full-vocab KL.'},ensure_ascii=False,indent=2))
    rows=[]
    with (a.out/'responses.jsonl').open('x') as f:
        for batch in plan:
            barrier=threading.Barrier(len(batch))
            def send(case):
                row=copy.deepcopy(case);barrier.wait();row['start_ns']=time.time_ns()
                try:
                    row['response']=post(a.base_url,'/v1/chat/completions',case['request']);choice=row['response']['choices'][0];row['parsed']=json.loads(choice['message']['content']);row['correct']=choice['finish_reason']=='stop' and canonical(row['parsed'])==canonical(case['expected'])
                except Exception as e:row.update(correct=False,error=repr(e))
                row['end_ns']=time.time_ns();return row
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                for row in pool.map(send,batch):
                    rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(json.dumps({k:row[k] for k in ('repeat','concurrency','id','correct')}),flush=True)
    pairs=[]
    for repeat in range(3):
        left=next(r for r in rows if r['id']=='short-4' and r['repeat']==repeat and r['concurrency']==1);right=next(r for r in rows if r['id']=='short-4' and r['repeat']==repeat and r['concurrency']==3)
        x=left.get('response',{}).get('choices',[{}])[0];y=right.get('response',{}).get('choices',[{}])[0];tx=x.get('token_ids') or [];ty=y.get('token_ids') or [];i=next((i for i,(u,v) in enumerate(zip(tx,ty)) if u!=v),None)
        pair={'repeat':repeat,'C1_correct':left['correct'],'C3_correct':right['correct'],'token_ids_available':bool(tx and ty),'same_token_ids':tx==ty if tx and ty else None,'first_divergence':i}
        if i is not None:
            pair.update(common_prefix_token_ids=tx[:i],C1_token_id=tx[i],C3_token_id=ty[i],C1_logprobs=x['logprobs']['content'][i],C3_logprobs=y['logprobs']['content'][i],scope='Same generated token prefix until this position; raw model distribution before sampling processors')
        pairs.append(pair)
    summary={'requests':len(rows),'target_requests':6,'target_correct':sum(r['correct'] for r in rows if r['id']=='short-4'),'pairs':pairs,'requires_actual_batch_trace':True,'conclusion':'No automatic numerical-root-cause classification from logprobs or non-reproduction'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
