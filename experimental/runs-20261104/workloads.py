#!/usr/bin/env python3
"""Synthetic cold replay then bounded dual-rank profiler windows; never reads real text."""
import argparse,concurrent.futures,hashlib,json,pathlib,random,threading,time,urllib.request,uuid
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--only-remaining',action='store_true');p.add_argument('--profiles-only',action='store_true');a=p.parse_args();root=a.root.resolve();out=root/'experiment';base='http://127.0.0.1:18106';model='Qwen3.6-35B-A3B-w8a8'
lock=threading.Lock();log=out/'http-events.jsonl'
def emit(kind,**fields):
 with lock:
  with log.open('a') as f:f.write(json.dumps({'kind':kind,'wall':time.time(),'mono':time.monotonic(),**fields})+'\n')
 print(kind,{k:v for k,v in fields.items() if k not in ('usage',)},flush=True)
def post(path,payload,timeout=1200):
 data=json.dumps(payload).encode();return urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'}),timeout=timeout)
def control(path):
 with post(path,{}) as response:return response.read().decode()
def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as r:text=r.read().decode()
 counters={}
 for line in text.splitlines():
  for name in ['prefix_cache_hits_total','prefix_cache_queries_total']:
   if line.startswith('vllm:'+name+'{'):counters[name]=counters.get(name,0)+float(line.split()[-1])
 return counters
with post('/tokenize',{'model':model,'prompt':'会议记录：项目进度、风险和行动项。 Alice reviews the project schedule and decisions. ','add_special_tokens':False}) as response:unit=json.load(response)['tokens']
assert unit

def request(label,prompt,output,first_event=None):
 tokens=(unit*((prompt+len(unit)-1)//len(unit)))[:prompt]
 begin=time.monotonic();emit('request_start',label=label,prompt_tokens=prompt,output_tokens=output)
 first=None;last=None;usage=None;identified=False
 payload={'model':model,'prompt':tokens,'max_tokens':output,'temperature':0,'seed':17,'ignore_eos':True,'cache_salt':uuid.uuid4().hex,'stream':True,'stream_options':{'include_usage':True}}
 with post('/v1/completions',payload) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   event=json.loads(raw)
   if not identified and event.get('id'):
    identified=True
    emit('request_identity',label=label,ids=[hashlib.sha256((event['id']+suffix).encode()).hexdigest()[:20] for suffix in ('','-0')])
   if any(choice.get('text') for choice in event.get('choices',[])):
    last=time.monotonic()
    if first is None:
     first=last;emit('first_output',label=label,elapsed=first-begin)
     if first_event:first_event.set()
   if event.get('usage'):usage=event['usage']
 if first_event:first_event.set()
 assert usage and usage['prompt_tokens']==prompt and usage['completion_tokens']==output,(label,usage)
 emit('request_end',label=label,start=begin,first=first,last=last,end=time.monotonic(),usage=usage)
 return {'label':label,'prompt':prompt,'output':output,'start':begin,'first':first,'last':last,'end':time.monotonic()}

# Busy-time conditioning is explicit: site histories are mostly idle. seqs4 caps observed >4.
if not (a.only_remaining or a.profiles_only):
 lengths=json.loads((root/'paired-lengths-arrivals.json').read_text());rng=random.Random(20261104)
 ordered=sorted(lengths,key=lambda r:r['input'])
 # Stratified joint empirical samples, preserving each sampled input/output pair.
 pairs=[ordered[min(len(ordered)-1,int((i+.5)*len(ordered)/18))] for i in range(18)]
 rng.shuffle(pairs)
 # A 600-second admission window, concurrency targets from observed busy-time histogram.
 # End admissions at 480 sec and drain in-flight requests; output token lengths are never clipped.
 schedule=[(0,1),(285,2),(330,3),(365,4),(480,0)]
 manifest={'kind':'synthetic_busy_time_replay','target_seconds':600,'admission_stop_seconds':480,
           'source':'empirical paired lengths under input>=4096/output>=128 proxy',
           'arrival_model':'closed-loop occupancy targets from visible-request busy-time histogram; arrival times are generated, not exact production replay',
           'schedule':schedule,'seed':20261104,'pairs':[{'input':r['input'],'output':r['output']} for r in pairs],
           'prefix':'enabled; independent salt per request; cold first request; no generation warmup'}
 (out/'workload-manifest.json').write_text(json.dumps(manifest,indent=2))
 before=metrics();emit('replay_start',prefix=before)
 started=time.monotonic();completed=[];futures={};index=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  while True:
   elapsed=time.monotonic()-started
   for future in list(futures):
    if future.done():completed.append(future.result());del futures[future]
   target=next((n for t,n in reversed(schedule) if elapsed>=t),1)
   while len(futures)<target and elapsed<480:
    pair=pairs[index%len(pairs)];label='replay-%03d'%index;index+=1
    future=pool.submit(request,label,pair['input'],pair['output']);futures[future]=label
   if elapsed>=480 and not futures:break
   if elapsed>1000:raise TimeoutError('Cold replay exceeded bounded 1000-second drain budget')
   time.sleep(.1)
 after=metrics();emit('replay_end',elapsed=time.monotonic()-started,requests=len(completed),prefix=after)
 (out/'replay-requests.json').write_text(json.dumps(completed,indent=2))

class StepReader:
 def __init__(self):self.positions={}
 def read(self):
  rows=[]
  for path in (out/'observation').glob('observe-*.jsonl'):
   with path.open() as f:
    f.seek(self.positions.get(str(path),0))
    while True:
     pos=f.tell();line=f.readline()
     if not line:break
     if not line.endswith('\n'):f.seek(pos);break
     row=json.loads(line)
     if row['kind']=='forward_start':rows.append(row)
    self.positions[str(path)]=f.tell()
  return rows
reader=StepReader();reader.read()

def profile(label,batch,prompt,mixed=False,mapping=False):
 emit('profile_prepare',label=label,batch=batch,prompt_tokens=prompt)
 first=[threading.Event() for _ in range(batch)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=batch+1) as pool:
  tasks=[pool.submit(request,label+'-%d'%i,prompt,320 if mixed else 256,first[i]) for i in range(batch)]
  for event in first:
   if not event.wait(180):raise TimeoutError(label+' first-output barrier')
  reader.read()
  start=time.monotonic();control('/start_profile');emit('profile_start',label=label,batch=batch)
  if mapping:(out/'observation'/'map.enabled').touch()
  counts={};stable_counts={}
  if mixed:
   tasks.append(pool.submit(request,label+'-prefill',32768,256))
  try:
   while True:
    for row in reader.read():
     if row['mono']<start or row['capturing']:continue
     pid=row['pid'];counts[pid]=counts.get(pid,0)+1
     eligible=(row['batch']==(4 if mixed else batch) and row['padded_tokens']==row['batch'])
     if eligible:stable_counts[pid]=stable_counts.get(pid,0)+1
    if stable_counts and min(stable_counts.values())>=112:break
    if time.monotonic()-start>180:raise TimeoutError(label+' profiler stable-step budget')
    if all(t.done() for t in tasks):break
    time.sleep(.04)
  finally:
   control('/stop_profile');emit('profile_stop',label=label,observed_steps=counts,stable_steps=stable_counts,elapsed=time.monotonic()-start)
   (out/'observation'/'map.enabled').unlink(missing_ok=True)
  for task in tasks:task.result()
 emit('profile_done',label=label)

if not a.only_remaining:
 profile('b1-long',1,32768)
 profile('b2-long',2,32768)
profile('b3-short',3,4096,mapping=not a.profiles_only)
# Fourth admission creates a chunked-prefill/decode mixed window and a stable B4 tail.
profile('mixed-b4',3,4096,mixed=True)
emit('all_done')
