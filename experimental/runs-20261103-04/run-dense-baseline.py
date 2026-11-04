#!/usr/bin/env python3
"""在a2独占建立27B Dense原始算子基线；原始模型/OPP只读，结束恢复临时unit。"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--diagnostic-sync',action='store_true',help='startup diagnostic only; no performance measurements')
p.add_argument('--disable-executor-cache',action='store_true',help='CANN executor cache limit 0; separate runtime compatibility screen')
p.add_argument('--startup-only',action='store_true',help='health and short checks only, no performance benchmark')
p.add_argument('--quant-trace',action='store_true',help='capture actual down_proj inputs; implies synchronous eager diagnostic')
p.add_argument('--disable-expandable',action='store_true',help='original production image allocator control only')
p.add_argument('--quant-repack',action='store_true',help='rebuild down_proj weight storage with strict value gate; synchronous eager only')
p.add_argument('--runtime-compatibility',action='store_true',help='separate audited post4 runtime startup screen')
p.add_argument('--gpu-memory-utilization',type=float,choices=[0.90,0.92,0.94],help='Dense compatibility capacity control; context remains unchanged')
p.add_argument('--attempt',default='',help='unique suffix preserving prior failed experiments')
a=p.parse_args()
if a.attempt and not re.fullmatch(r'[A-Za-z0-9-]+',a.attempt):
 p.error('attempt must contain only ASCII letters, digits and hyphens')
if a.disable_expandable and (a.runtime_compatibility or a.quant_trace or a.quant_repack or a.disable_executor_cache):
 p.error('--disable-expandable must be isolated from other runtime interventions')
if a.gpu_memory_utilization is not None and not a.runtime_compatibility:
 p.error('Dense capacity control is restricted to the separate compatibility screen')
if a.runtime_compatibility and (not a.startup_only or a.quant_trace or a.quant_repack or a.disable_executor_cache):
 p.error('--runtime-compatibility requires --startup-only and no kernel/tensor interventions')
if a.quant_repack and (a.quant_trace or a.disable_executor_cache):
 p.error('--quant-repack cannot be combined with other runtime interventions')
a.diagnostic_sync=a.diagnostic_sync or a.quant_trace or a.quant_repack;root=a.root.resolve();suffix='-no-expandable' if a.disable_expandable else '-post4-runtime-screen' if a.runtime_compatibility else '-quant-repack' if a.quant_repack else '-no-executor-cache-screen' if a.disable_executor_cache else '-quant-trace' if a.quant_trace else '-sync-eager-diagnostic' if a.diagnostic_sync else '';suffix+=('-'+a.attempt) if a.attempt else '';out=root/('qwen38-dense-baseline'+suffix);out.mkdir(exist_ok=False)
container='qwen38-dense-baseline'+suffix+'-20261103';port=18111 if a.diagnostic_sync else 18110;base=f'http://127.0.0.1:{port}'
model='Qwen3.8-27B-w8a8-310p';common=['--base-url',base,'--model',model]
shutil.copy2(__file__,out/'supervisor-used.py')
steps=[];paused=False;launched=False;finished=False;primary_error=None

def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
def run(label,cmd):
 state('running',current=label);begin=time.monotonic()
 with (out/(label+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-begin))
 if r.returncode:raise RuntimeError(label+' failed')
def quality(label):
 run(label,['python3',str(root/'check_cache_request_reuse.py'),*common,'--out',str(out/(label+'.json'))])
 j=json.loads((out/(label+'.json')).read_text())
 if j['strict_matches']!=8 or not j['reuse']['identical_text'] or j['reuse']['canary_leaked']:raise RuntimeError(label+' failed strict checks')
try:
 state('waiting_operator_control')
 if a.disable_executor_cache:
  while True:
   f=root/'executor-cache-control/status.json';j=json.loads(f.read_text()) if f.exists() else {}
   if j.get('state')=='complete':break
   if j.get('state')=='failed':
    if j.get('error')=='Could not restore a2 temporary unit' and len(j.get('steps',[]))==2 and all(step['returncode']==0 for step in j['steps']):break
    raise RuntimeError('Cache control failed before model trial')
   time.sleep(5)
 if a.quant_trace:
  while True:
   f=root/'upstream-fwd-o/status.json';j=json.loads(f.read_text()) if f.exists() else {}
   if j.get('state') in ('passed_operator_validation','failed') and (f.parent/'restore-a2-service.log').exists():break
   time.sleep(5)
 while True:
  f=root/'fwd-o-rebuild-control/status.json'
  j=json.loads(f.read_text()) if f.exists() else {}
  if j.get('state')=='complete':break
  if j.get('state')=='failed':raise RuntimeError('Operator control failed; avoid overlapping recovery')
  time.sleep(5)
 downloaded=json.loads((root/'qwen38-27b-w8a8-310p/status.json').read_text())
 if downloaded['state']!='complete':raise RuntimeError('Weights not verified')
 manifest=json.loads((root/'production-manifest.json').read_text())
 command=manifest['command'];command[command.index('serve')+1]='/models/'+model
 command[command.index('--served-model-name')+1]=model
 if a.quant_trace or a.quant_repack:
  tools=out/'tools';tools.mkdir()
  worker='quant_repack_worker' if a.quant_repack else 'quant_trace_worker'
  shutil.copy2(root/(worker+'.py'),tools/(worker+'.py'))
  manifest['diagnostic_worker_sha256']=hashlib.sha256((tools/(worker+'.py')).read_bytes()).hexdigest()
  command+=['--worker-cls',worker+('.QuantRepackWorker' if a.quant_repack else '.QuantTraceWorker')]
 if a.runtime_compatibility:
  manifest['image']='b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d'
  manifest['experiment_kind']='runtime_compatibility_screen'
 if a.gpu_memory_utilization is not None:
  command[command.index('--gpu-memory-utilization')+1]=str(a.gpu_memory_utilization)
  manifest['capacity_control']={'gpu_memory_utilization':a.gpu_memory_utilization,'max_model_len':262144}
 manifest['model_name']=model
 manifest['experiment_scope']='Qwen3.8-27B Dense initial TP2 W8A8 baseline on a2; production image/config reused, model differs; no a1 performance equivalence claim'
 manifest['env']['PYTHONFAULTHANDLER']='1'
 if a.disable_expandable:
  manifest['env']['PYTORCH_NPU_ALLOC_CONF']='expandable_segments:False'
  manifest['allocator_control']='expandable_segments:False; original production image and unmodified operators'
 if a.disable_executor_cache:
  manifest['env']['ACLNN_CACHE_LIMIT']='0'
  manifest['experiment_scope']='Dense runtime compatibility screen, only ACLNN_CACHE_LIMIT=0 differs from original startup; no performance acceptance'
 if a.diagnostic_sync:
  manifest['env']['ASCEND_LAUNCH_BLOCKING']='1'
  command[command.index('--compilation-config')+1]=json.dumps(dict(mode=0,cudagraph_mode='NONE'))
  command.append('--enforce-eager')
  manifest['experiment_scope']='Dense synchronous startup diagnostic; performance measurements prohibited'
 if a.runtime_compatibility:
  manifest['experiment_scope']='Entire audited post4 image compatibility screen; synchronous eager if requested; no production A/B or performance acceptance'
 mf=out/'dense-manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n')
 paused=True;run('pause-a2-temporary-unit',['sudo','-n','systemctl','stop','llm-backend-primary.service'])
 idle=0
 for _ in range(90):
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  idle=idle+1 if 'No running processes found' in info.stdout else 0
  if idle>=2:break
  time.sleep(2)
 else:raise RuntimeError('NPU not idle')
 launched=True
 run('launch',['python3',str(root/'launch-production-replica.py'),'--manifest',str(mf),
  '--start-script',str(root/('start-quant-trace.sh' if a.quant_trace or a.quant_repack else 'start-production-replica.sh')),'--model-dir',str(root/'qwen38-27b-w8a8-310p/model'),
  '--out',str(out),'--container',container,'--port',str(port)]+(['--compatibility-image-id',manifest['image']] if a.runtime_compatibility else []))
 state('waiting_health')
 for _ in range(240):
  try:
   with urllib.request.urlopen(base+'/health',timeout=3) as response:
    if response.status==200:break
  except Exception:pass
  status=subprocess.run(['podman','inspect','--format','{{.State.Status}}',container],capture_output=True,text=True)
  if status.stdout.strip()=='exited':raise RuntimeError('Dense service exited during startup')
  time.sleep(5)
 else:raise TimeoutError('Dense health')
 quality('quality-before')
 if not a.diagnostic_sync and not a.startup_only:
  for length in (1024,4096,16384,32768):
   label=f'c1-p{length}-o256'
   run(label,['python3',str(root/'bench_server_phases.py'),*common,'--prompt-tokens',str(length),'--output-tokens','256',
    '--repeats','3','--seed','0','--cache-salt-prefix',f'dense-baseline-{length}','--out',str(out/(label+'.jsonl'))])
   rows=[r for r in map(json.loads,(out/(label+'.jsonl')).read_text().splitlines()) if r['kind']=='request']
   if len(rows)!=3 or any(not r['valid'] for r in rows):raise RuntimeError('Invalid phase measurements: '+label)
  run('business-facts',['python3',str(root/'check_business_facts.py'),*common,'--cases','8','--concurrencies','1','3','--out',str(out/'business-facts.jsonl')])
  run('business-quality',['python3',str(root/'check_business_quality.py'),'--url',base,'--model',model,
   '--fixtures-dir','/path/to/lab-runtime/a1-310p-tune-20261101','--out',str(out/'business-quality.jsonl')])
  run('gsm64',['python3',str(root/'eval_gsm8k_http.py'),*common,
   '--dataset','/path/to/lab-runtime/qwen36-public-release-20261012/gsm8k-test.jsonl','--limit','64','--seed','310',
   '--max-tokens','512','--label','qwen38-dense-baseline','--out',str(out/'gsm64.jsonl')])
 quality('quality-after');finished=True
except Exception as error:
 primary_error=str(error);state('failed',error=primary_error);raise
finally:
 if launched:
  stop=subprocess.run(['podman','stop','--time','90',container],capture_output=True,text=True)
  with (out/'server.log').open('w') as log:subprocess.run(['podman','logs',container],stdout=log,stderr=subprocess.STDOUT)
  status=subprocess.run(['podman','inspect','--format','{{.State.Status}}',container],capture_output=True,text=True)
  if status.stdout.strip()!='exited':finished=False;state('failed',error='Dense container cleanup incomplete',primary_error=primary_error)
 if paused:
  with (out/'restore-a2-service.log').open('x') as log:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit',primary_error=primary_error)
 if finished:state('complete',scope='Startup-only compatibility screen; no performance data' if a.startup_only else 'Synchronous startup diagnostic; no performance data' if a.diagnostic_sync else 'Initial Dense baseline; automatic quality checks require semantic review before acceptance')
