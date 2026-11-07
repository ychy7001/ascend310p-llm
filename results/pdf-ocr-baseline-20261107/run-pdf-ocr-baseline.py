"""One isolated Torch-WY baseline OCR smoke window; no candidate/gate changes."""
import argparse,base64,fcntl,hashlib,importlib.util,json,signal,subprocess,sys,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('root','tools','inputs','out'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();a.out.mkdir(exist_ok=False);sys.path.insert(0,str(a.tools))
spec=importlib.util.spec_from_file_location('ocr_launch_template',a.tools/'run-wy-model-gate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
 with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def state(value,**details):
 q=a.out/'status.tmp';q.write_text(json.dumps({'state':value,'unix_ns':time.time_ns(),**details},indent=2));q.replace(a.out/'status.json')
def snapshot():
 commands={'rootless':['podman','ps','-q'],'rootful':['sudo','-n','podman','ps','-q'],'primary':['systemctl','is-active','llm-backend-primary.service'],'image':['systemctl','--user','is-active','qwen36-image.service'],'npu':['npu-smi','info'],'memory':['free','-g']}
 return {k:{'code':v.returncode,'stdout':v.stdout.strip()} for k,c in commands.items() for v in [subprocess.run(c,capture_output=True,text=True,timeout=30)]}
class Launch(m.Gate):
 def state(self,value,**kw):state(value,**kw)
 def run(self,label,cmd):
  if label.endswith('-launch'):
   manifest_path=self.out/'torch'/'manifest.json';manifest=json.loads(manifest_path.read_text());command=manifest['command']
   command[command.index('--worker-cls')+1]='cann921_gdn_pair_worker.CANN921GDNPairWorker'
   manifest['env'].pop('WY_GATE_ARM',None)
   manifest_path.write_text(json.dumps(manifest,indent=2))
   write(self.out/'launch-manifest-identity.json',{'sha':sha(manifest_path),'worker':'CANN921GDNPairWorker, original Torch WY','image':manifest['image']})
  write(self.out/(label+'-command.json'),cmd)
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
  (self.out/(label+'.log')).write_text(result.stdout+result.stderr)
  result.check_returncode()
 def stop(self,label):
  if self.container is None:return
  name=self.container
  subprocess.run(['podman','stop','--time','30',name],capture_output=True,check=True,timeout=60)
  with (self.out/'torch-server.log').open('w') as f:subprocess.run(['podman','logs',name],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
  detail=json.loads(subprocess.check_output(['podman','inspect',name],text=True,timeout=30));write(self.out/'torch-container-inspect.json',detail)
  self.container=None
  assert not detail[0]['State']['Running'] and not detail[0]['State']['OOMKilled']
manifest=json.loads((a.inputs/'manifest.json').read_text());assert len(manifest['cases'])==5 and manifest['manual_ground_truth'] is False
for c in manifest['cases']:assert sha(a.inputs/c['image'])==c['image_sha256']
write(a.out/'freeze.json',{'controller_sha':sha(Path(__file__)),'launch_template_sha':sha(a.tools/'run-wy-model-gate.py'),'paired_worker_sha':sha(a.root/'upstream-gdn-pair921-v1/tools/cann921_gdn_pair_worker.py'),'input_manifest_sha':sha(a.inputs/'manifest.json'),'source_manifest_sha':sha(a.root/'production-manifest.json'),'case_ids':[c['id'] for c in manifest['cases']],'prompt':manifest['prompt'],'temperature':0,'seed':0,'max_tokens':3072,'thinking':False,'scope':'5-page Torch-WY OCR smoke, no quality/performance acceptance; no native candidate','action_budget_s':3000})
x=Launch.__new__(Launch);x.root=a.root;x.tools=a.tools;x.out=a.out;x.container=None;x.steps=[];x.started=time.monotonic();x.manifest=json.loads((a.root/'production-manifest.json').read_text())
error=None;started=None;results=[]
with open('/tmp/a310-npu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();start_ns=time.time_ns()
 def interrupted(*_):raise TimeoutError('OCR smoke bounded window interrupted')
 signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted);signal.alarm(3000)
 try:
  before=snapshot();write(a.out/'window-start.json',before)
  assert before['rootless']['stdout']==before['rootful']['stdout']==''
  assert before['primary']['stdout']==before['image']['stdout']=='inactive'
  x.launch('torch','A')
  for c in manifest['cases']:
   state('ocr_request',case=c['id'],completed=len(results))
   payload={'model':m.MODEL,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((a.inputs/c['image']).read_bytes()).decode()}},{'type':'text','text':manifest['prompt']}]}],'temperature':0,'seed':0,'max_tokens':3072,'chat_template_kwargs':{'enable_thinking':False},'cache_salt':'ocr-pdf-20261107-'+c['id'],'stream':True,'stream_options':{'include_usage':True}}
   encoded=json.dumps(payload).encode();write(a.out/(c['id']+'-request-identity.json'),{'payload_sha':hashlib.sha256(encoded).hexdigest(),'image_sha':c['image_sha256'],'max_tokens':3072})
   request=urllib.request.Request(m.BASE+'/v1/chat/completions',data=encoded,headers={'Content-Type':'application/json'})
   t=time.monotonic();first=None;first_content=None;text=[];reason=[];usage=None;finish=None
   with urllib.request.urlopen(request,timeout=600) as response,(a.out/(c['id']+'-stream.jsonl')).open('x') as raw:
    for line in response:
     if not line.startswith(b'data: '):continue
     body=line[6:].strip()
     if body==b'[DONE]':break
     chunk=json.loads(body);elapsed=time.monotonic()-t;raw.write(json.dumps({'elapsed_s':elapsed,'chunk':chunk},ensure_ascii=False)+'\n');raw.flush()
     if chunk.get('usage'):usage=chunk['usage']
     for choice in chunk.get('choices',[]):
      delta=choice.get('delta',{});content=delta.get('content') or '';thinking=delta.get('reasoning') or delta.get('reasoning_content') or ''
      if first is None and (content or thinking):first=elapsed
      if first_content is None and content:first_content=elapsed
      text.append(content);reason.append(thinking)
      if choice.get('finish_reason'):finish=choice['finish_reason']
   result={'id':c['id'],'first_delta_s':first,'first_content_s':first_content,'elapsed_s':time.monotonic()-t,'usage':usage,'finish_reason':finish,'text':''.join(text),'reasoning':''.join(reason),'image_sha':c['image_sha256']}
   write(a.out/(c['id']+'-response.json'),result);assert result['text'] and finish in ('stop','length'),result
   results.append({k:v for k,v in result.items() if k not in ('text','reasoning')});write(a.out/(c['id']+'-metrics.json'),results[-1])
  write(a.out/'summary.json',{'scope':'Torch baseline OCR smoke only, PDF text references not manual labels','completed':len(results),'results':results})
 except BaseException as exc:
  error=repr(exc)
  if hasattr(exc,'read'):(a.out/'http-error.txt').write_bytes(exc.read())
 finally:
  signal.alarm(0)
  try:x.stop('torch')
  finally:
   after=snapshot();clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and after['primary']['stdout']==after['image']['stdout']=='inactive' and 'No running processes' in after['npu']['stdout']
   write(a.out/'window-closed.json',{'start_unix_ns':start_ns,'end_unix_ns':time.time_ns(),'elapsed_s':time.monotonic()-started,'cleanup_passed':clean,'snapshot':after,'error':error})
   state('failed' if error else 'complete',error=error,completed=len(results));assert clean
if error:raise RuntimeError(error)
