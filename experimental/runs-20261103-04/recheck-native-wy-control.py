#!/usr/bin/env python3
"""B 通过后重测原配置 A'，并独占复核数值异常，排除跨时段漂移。"""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();native=root/'native-gdn';b=root/'production-native-wy'
out=root/'production-wy-control-recheck';out.mkdir(exist_ok=False)
container='qwen36-wy-control-recheck-20261103';base='http://127.0.0.1:18109'
common=['--base-url',base,'--model','Qwen3.6-35B-A3B-w8a8'];steps=[];paused=False;launched=False;finished=False;brecheck_started=False

def state(value,**kw):
 tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');tmp.replace(out/'status.json')
def run(label,cmd):
 state('running',current=label);start=time.monotonic()
 with (out/(label+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-start))
 if r.returncode:raise RuntimeError(label+' failed')
def idle():
 count=0
 for _ in range(90):
  r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  count=count+1 if 'No running processes found' in r.stdout else 0
  if count>=2:return
  time.sleep(2)
 raise RuntimeError('NPU cleanup incomplete')
def quality(label):
 run(label,['python3',str(root/'check_cache_request_reuse.py'),*common,'--out',str(out/(label+'.json'))])
 j=json.loads((out/(label+'.json')).read_text())
 if j['strict_matches']!=8 or not j['reuse']['identical_text'] or j['reuse']['canary_leaked']:raise RuntimeError('short gate')
def rows(path,kind):return [r for r in map(json.loads,path.read_text().splitlines()) if r['kind']==kind]
try:
 state('waiting_candidate')
 for _ in range(1440):
  j=json.loads((b/'status.json').read_text())
  if j['state']=='failed':state('not_started',reason='candidate failed');raise SystemExit(0)
  if j['state']=='complete':break
  time.sleep(5)
 else:raise TimeoutError('candidate')
 if j['decision']!='promising_requires_repeat_and_semantic_review':
  state('not_started',reason='candidate did not pass screening/quality');raise SystemExit(0)
 # Candidate controller restores the original unit before writing complete.
 # User explicitly designated a2 as an experiment host; stop only this temporary unit.
 paused=True
 run('pause-a2-temporary-unit',['sudo','-n','systemctl','stop','llm-backend-primary.service'])
 state('waiting_devices_idle');idle()
 # Repeat the previously shared-device diagnostic on an idle device.
 cmd=json.loads((native/'validation-command.json').read_text())
 cmd[cmd.index('--name')+1]='qwen36-native-wy-isolated-order-20261103'
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="/work/opp/vendors/hyperqwen_wy_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
cd /work/tools
python3 probe_native_wy_order.py --work /work --out-dir /work/isolated-order
'''
 run('isolated-numerical-order',cmd)
 result=[json.loads(x) for x in (native/'isolated-order/order-diagnostic.jsonl').read_text().splitlines()]
 summary={mode:dict(tested=sum(x['mode']==mode for x in result),within_tolerance=sum(x['mode']==mode and x['within_tolerance'] for x in result)) for mode in ('torch','native')}
 (out/'isolated-numerical-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 if summary['native']!={'tested':12,'within_tolerance':12}:raise RuntimeError('Candidate failed isolated numerical repeat')
 state('waiting_devices_idle');idle()
 brecheck_started=True
 run('restart-candidate-for-semantics',['podman','start','qwen36-production-native-wy-20261103'])
 state('waiting_candidate_semantic_health')
 for _ in range(240):
  try:
   with urllib.request.urlopen('http://127.0.0.1:18108/health',timeout=3) as r:
    if r.status==200:break
  except Exception:pass
  r=subprocess.run(['podman','inspect','--format','{{.State.Status}}','qwen36-production-native-wy-20261103'],capture_output=True,text=True)
  if r.stdout.strip()=='exited':raise RuntimeError('Candidate semantic restart failed')
  time.sleep(5)
 else:raise TimeoutError('candidate semantic health')
 run('candidate-meeting-owners',['python3',str(root/'recheck_meeting_owners.py'),
     '--base-url','http://127.0.0.1:18108','--reference-jsonl',str(root/'production-w8a8/business-quality.jsonl'),
     '--out',str(out/'candidate-meeting-owners.jsonl')])
 run('stop-semantic-candidate',['podman','stop','--time','90','qwen36-production-native-wy-20261103'])
 brecheck_started=False
 state('waiting_devices_idle');idle()
 manifest=json.loads((root/'production-manifest.json').read_text());manifest['env']['PYTHONFAULTHANDLER']='1'
 mf=out/'control-manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n')
 launched=True
 run('launch',['python3',str(root/'launch-production-replica.py'),'--manifest',str(mf),
      '--start-script',str(root/'start-production-replica.sh'),'--model-dir',str(root/'production-w8a8/model'),
      '--out',str(out),'--container',container,'--port','18109','--profiler'])
 state('waiting_health')
 for _ in range(240):
  try:
   with urllib.request.urlopen(base+'/health',timeout=3) as r:
    if r.status==200:break
  except Exception:pass
  r=subprocess.run(['podman','inspect','--format','{{.State.Status}}',container],capture_output=True,text=True)
  if r.stdout.strip()=='exited':raise RuntimeError('Control exited during startup')
  time.sleep(5)
 else:raise TimeoutError('control health')
 quality('quality-before')
 comparison={}
 for n in (1024,4096,16384,32768):
  label=f'c1-p{n}-o256'
  run(label,['python3',str(root/'bench_server_phases.py'),*common,'--prompt-tokens',str(n),
      '--output-tokens','256','--repeats','3','--seed','0','--cache-salt-prefix',f'wy-control-{n}',
      '--out',str(out/(label+'.jsonl'))])
  control=rows(out/(label+'.jsonl'),'request');candidate=rows(b/(label+'.jsonl'),'request')
  original=rows(root/'production-w8a8'/(label+'.jsonl'),'request')
  if len(control)!=3 or not all(r['valid'] for r in control):raise RuntimeError('Invalid control requests')
  if [r['prompt_ids_sha256'] for r in control]!=[r['prompt_ids_sha256'] for r in candidate]:raise RuntimeError('Prompt mismatch')
  comparison[str(n)]={}
  for key in ('prefill_tok_s','decode_tok_s'):
   c=statistics.median(r[key] for r in control);v=statistics.median(r[key] for r in candidate);old=statistics.median(r[key] for r in original)
   comparison[str(n)][key]=dict(control=c,candidate=v,original=old,candidate_over_control=v/c,control_over_original=c/old)
 (out/'phase-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
 for c in (1,3,4):
  label=f'c{c}-p1024-o256-stream'
  run(label,['python3',str(root/'bench_http_stream.py'),*common,'--prompt-tokens','1024','--output-tokens','256',
      '--repeats','3','--seed','0','--concurrency',str(c),'--out',str(out/(label+'.jsonl'))])
 run('control-business-quality',['python3',str(root/'check_business_quality.py'),'--url',base,'--model',common[-1],
     '--fixtures-dir','/path/to/lab-runtime/a1-310p-tune-20261101','--out',str(out/'business-quality.jsonl')])
 run('control-meeting-owners',['python3',str(root/'recheck_meeting_owners.py'),*common[:2],
     '--reference-jsonl',str(root/'production-w8a8/business-quality.jsonl'),
     '--out',str(out/'control-meeting-owners.jsonl')])
 fixtures=Path('/path/to/lab-runtime/a1-310p-tune-20261101')
 shutil.copy2(root/'production-w8a8/replay-prompts.json',out/'replay-prompts.json')
 run('business-replay-long',['python3',str(root/'replay_stage_workload.py'),'--workload',str(fixtures/'a1-310p-tune.workload.json'),
     '--recording','chinese_company_annual_meeting_planning','--urls',base,'--model',common[-1],
     '--prompts',str(out/'replay-prompts.json'),'--out',str(out/'business-replay-long.json')])
 quality('quality-after')
 finished=True;state('finishing')
except Exception as e:
 state('failed',error=str(e));raise
finally:
 if brecheck_started:
  subprocess.run(['podman','stop','--time','90','qwen36-production-native-wy-20261103'],capture_output=True,text=True)
 if launched:
  subprocess.run(['podman','stop','--time','90',container],capture_output=True,text=True)
  with (out/'server.log').open('w') as f:subprocess.run(['podman','logs',container],stdout=f,stderr=subprocess.STDOUT)
  r=subprocess.run(['podman','inspect','--format','{{.State.Status}}',container],capture_output=True,text=True)
  if r.stdout.strip()!='exited':finished=False;state('failed',error='Control cleanup incomplete')
 if paused:
  with (out/'restore-a2-service.log').open('x') as f:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary service')
 if finished:state('complete',scope='A/B/A phase, stream and business recheck; semantic review still required')
