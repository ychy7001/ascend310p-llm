"""Fixed five-point A/AP/B screen only; no full G1/G2/performance."""
import argparse,fcntl,hashlib,importlib.util,json,signal,subprocess,sys,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('root','tools','wy','out','previous','policy','R','g0'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.tools))
spec=importlib.util.spec_from_file_location('gate_template',a.tools/'run-wy-model-gate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def state(value,**kw):
 q=a.out/'status.tmp';q.write_text(json.dumps({'state':value,'unix_ns':time.time_ns(),**kw},indent=2));q.replace(a.out/'status.json')
def snapshot():
 commands={'rootless':['podman','ps','-q'],'rootful':['sudo','-n','podman','ps','-q'],'primary':['systemctl','is-active','llm-backend-primary.service'],'image':['systemctl','--user','is-active','qwen36-image.service'],'npu':['npu-smi','info'],'memory':['free','-g']}
 return {k:{'code':v.returncode,'stdout':v.stdout.strip()} for k,c in commands.items() for v in [subprocess.run(c,capture_output=True,text=True,timeout=30)]}
class Launch(m.Gate):
 def state(self,value,**kw):state(value,**kw)
 def run(self,label,cmd):
  if label.endswith('-launch'):
   source=str(self.root/'native-wy921-v1')+':/native-wy:ro';assert source in cmd;cmd[cmd.index(source)]=str(a.wy)+':/native-wy:ro'
  write(a.out/(label+'-command.json'),cmd)
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(a.out/(label+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
 def stop(self,label):
  if self.container is None:return
  name=self.container;subprocess.run(['podman','stop','--time','30',name],check=True,capture_output=True,timeout=60)
  with (a.out/(label+'-server.log')).open('x') as f:subprocess.run(['podman','logs',name],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
  detail=json.loads(subprocess.check_output(['podman','inspect',name],text=True,timeout=30));write(a.out/(label+'-container-inspect.json'),detail);self.container=None
  assert detail[0]['State']['ExitCode']==0 and not detail[0]['State']['OOMKilled'] and not detail[0]['State']['Running']
policy=read(a.policy);assert len(policy['points'])==5 and policy['scope']=='fixed_screen_only'
assert [(v['case'],v['position']) for v in policy['points'] if v['classification']=='structural']==[('replay-03',2)]
assert sha(a.wy/'source/csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/compute_wy_kernel.h')==policy['candidate_kernel_sha']
g0=read(a.g0);assert g0['passed'] and g0['wy_cases']==612 and g0['candidate_manifest_sha256']==sha(a.wy/'operator-artifacts.json')
assert sha(a.R)==policy['R_sha256'];protocol=read(a.previous/'protocol.json');selection=protocol['cpu_subset_protocol'];assert sha(a.previous/'inputs.json')==selection['input_sha256']
refs={c['id']:c['token_ids'] for c in read(a.previous/'controls/A-continuation.json')['cases']};cases={c['id']:c for c in read(a.previous/'inputs.json')['cases']}
for point in policy['points']:
 context=cases[point['case']]['token_ids']+refs[point['case']][:point['position']]
 assert hashlib.sha256(json.dumps(context,separators=(',',':')).encode()).hexdigest()==point['context_sha256']
a.out.mkdir(exist_ok=False)
write(a.out/'freeze.json',{'controller_sha':sha(Path(__file__)),'policy_sha':sha(a.policy),'g0_sha':sha(a.g0),'worker_sha':sha(a.tools/'wy_gate_worker.py'),'launch_template_sha':sha(a.tools/'run-wy-model-gate.py'),'start_script_sha':sha(a.tools/'start-wy-gate.sh'),'wy_manifest_sha':sha(a.wy/'operator-artifacts.json'),'protocol_sha':sha(a.previous/'protocol.json'),'rule_sha':protocol['cpu_subset_protocol_sha256'],'rule_source':'protocol.cpu_subset_protocol','input_sha':sha(a.previous/'inputs.json'),'continuation_sha':sha(a.previous/'controls/A-continuation.json'),'R_sha':sha(a.R),'scope':'fixed screen only; historical G1 fails preserved; no G2/performance/full G1','action_budget_s':3000})
x=Launch.__new__(Launch);x.root=a.root;x.tools=a.tools;x.out=a.out;x.container=None;x.steps=[];x.started=time.monotonic();x.manifest=read(a.root/'production-manifest.json')
error=None;records={};started=None;live_label=None
with open('/tmp/a310-npu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();start_ns=time.time_ns()
 def interrupted(*_):raise TimeoutError('fixed screen action deadline')
 signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted);signal.alarm(3000)
 try:
  before=snapshot();write(a.out/'window-start.json',before);assert before['rootless']['stdout']==before['rootful']['stdout']=='';assert before['primary']['stdout']==before['image']['stdout']=='inactive'
  for arm in ('A','A_prime','B'):
   live_label='screen-'+arm;x.launch(live_label,'A' if arm=='A_prime' else arm);raw=a.out/(live_label+'-raw');raw.mkdir();rows=[]
   for point in policy['points']:
    case=point['case'];pos=point['position'];context=cases[case]['token_ids']+refs[case][:pos]
    state('collecting',arm=arm,case=case,position=pos,completed=len(rows))
    payload={'model':m.MODEL,'prompt':context,'temperature':0,'seed':0,'max_tokens':1,'ignore_eos':True,'logprobs':20,'return_token_ids':True,'return_tokens_as_token_ids':True,'cache_salt':f'wy-postbeta-refined-screen-{arm}-{case}-{pos}'}
    request=urllib.request.Request(m.BASE+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:result=json.load(response)
    write(raw/(case+f'-p{pos:02d}.json'),result);choice=result['choices'][0];top=choice['logprobs']['top_logprobs'][0];assert len(top)==20 and len(choice['token_ids'])==1 and all(t.startswith('token_id:') for t in top)
    lp={k.removeprefix('token_id:'):v for k,v in top.items()};observed=choice['token_ids'][0]
    rows.append({**point,'top1':observed,'matches_R':observed==point['R_top1'],'logprobs':lp,'target':refs[case][pos],'target_logprob':lp.get(str(refs[case][pos])),'usage':result.get('usage')})
   write(a.out/(live_label+'-results.json'),{'arm':arm,'points':rows});records[arm]=rows;x.stop(live_label)
  checks={arm:all(row['matches_R'] for row in rows if row['classification']=='structural') for arm,rows in records.items()};controls=checks['A'] and checks['A_prime'];eligible=controls and checks['B']
  write(a.out/'decision.json',{'screen_only':True,'controls_structural_match_R':controls,'candidate_structural_match_R':checks['B'],'eligible_for_full_G1':eligible,'near_tie_policy':'record_only','censored_policy':'indeterminate','historical_G1_fail':'preserved','G1':'not_started','G2':'not_started','performance':'not_started','adopted':False})
 except BaseException as exc:error=repr(exc)
 finally:
  signal.alarm(0)
  try:x.stop(live_label or 'emergency')
  finally:
   after=snapshot();clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and after['primary']['stdout']==after['image']['stdout']=='inactive' and 'No running processes' in after['npu']['stdout']
   write(a.out/'window-closed.json',{'start_unix_ns':start_ns,'end_unix_ns':time.time_ns(),'elapsed_s':time.monotonic()-started,'cleanup_passed':clean,'snapshot':after,'error':error});state('failed' if error else 'complete',error=error,arms_completed=len(records));assert clean
if error:raise RuntimeError(error)
