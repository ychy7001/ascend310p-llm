#!/usr/bin/env python3
"""Run preserved Chinese minutes, Responses tools, and five JSON-schema cases."""
import argparse,concurrent.futures,json,pathlib,time,urllib.request
import jsonschema
p=argparse.ArgumentParser();p.add_argument('--url',help='Endpoint URL; provide --url or --urls explicitly');p.add_argument('--urls',nargs='+');p.add_argument('--model',default='Qwen3.6-35B-A3B-w8a8');p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--fixtures-dir',type=pathlib.Path,required=True);a=p.parse_args();root=a.fixtures_dir
if a.out.exists():raise SystemExit('Refusing to append to existing quality output')
a.out.parent.mkdir(parents=True,exist_ok=True)
def send(path,payload,url):
 req=urllib.request.Request(url+path,json.dumps(payload).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=900) as response:return json.load(response)
schema={'type':'object','properties':{'topic':{'type':'string'},'budget':{'type':'integer'},'approved':{'type':'boolean'},'actions':{'type':'array','items':{'type':'object','properties':{'owner':{'type':'string'},'task':{'type':'string'},'due_date':{'type':'string'}},'required':['owner','task','due_date'],'additionalProperties':False}}},'required':['topic','budget','approved','actions'],'additionalProperties':False}
cases=[]
for line in (root/'meeting-prompts.jsonl').read_text().splitlines():
 d=json.loads(line);i=d.pop('id');d['model']=a.model;cases.append(('meeting',i,'/v1/chat/completions',d))
for line in (root/'tool-prompts.jsonl').read_text().splitlines():
 d=json.loads(line);i=d['id'];tools=[{'type':'function',**t['function']} for t in d['tools']]
 payload={'model':a.model,'input':d['messages'],'tools':tools,'tool_choice':'auto','temperature':0,'max_output_tokens':512,'chat_template_kwargs':{'enable_thinking':False}}
 cases.append(('tool',i,'/v1/responses',payload))
for i in range(1,6):
 text=f'仅输出符合schema的JSON。会议议题是项目{i}预算，预算{i*13000}元，已获批准。李明应在2026-11-{i+10:02d}前提交采购清单。'
 payload={'model':a.model,'messages':[{'role':'user','content':text}],'temperature':0,'max_tokens':1024,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'meeting','strict':True,'schema':schema}}}
 cases.append(('schema',i,'/v1/chat/completions',payload))
def run(case):
 kind,i,path,payload=case;start=time.time();row={'kind':kind,'id':i,'request':payload,'path':path,'ok':False}
 try:
  url=(a.urls or [a.url])[(i-1)%len(a.urls or [a.url])];row['backend_url']=url;d=send(path,payload,url);row['response']=d
  if kind=='tool':
   calls=[x for x in d.get('output',[]) if x.get('type')=='function_call'];assert len(calls)==1,calls
   tool=calls[0];assert tool['name']=='create_meeting_action';args=json.loads(tool['arguments']);assert set(('task','owner','due_date'))<=args.keys();expected={1:('完成接口联调','李强','2026-11-08'),2:('提交预算评估','陈洁','2026-11-03'),3:('修复报表时区','刘洋','2026-11-04')}[i];assert (args['task'],args['owner'],args['due_date'])==expected,args;assert d.get('status')=='completed',d.get('status');row['parsed_arguments']=args
  else:
   choice=d['choices'][0];text=choice['message']['content'];assert choice['finish_reason']=='stop',choice['finish_reason'];assert text and '\ufffd' not in text
   if kind=='schema':
    parsed=json.loads(text);jsonschema.validate(parsed,schema)
    assert parsed['topic']==f'项目{i}预算' and parsed['budget']==i*13000 and parsed['approved'] is True,parsed
    assert any(action['owner']=='李明' and action['due_date']==f'2026-11-{i+10:02d}' and '采购清单' in action['task'] for action in parsed['actions']),parsed
  row['ok']=True
 except Exception as e:row['error']=str(e)
 row['elapsed_s']=time.time()-start;return row
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for row in pool.map(run,cases):
  rows.append(row)
  with a.out.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  print(json.dumps({k:v for k,v in row.items() if k not in ('request','response')},ensure_ascii=False),flush=True)
print(json.dumps({'total':len(rows),'passed_auto':sum(x['ok'] for x in rows), 'scope':'meeting checks only format/completion; candidate summaries still require semantic comparison'}))
if not all(row['ok'] for row in rows):raise SystemExit(1)
