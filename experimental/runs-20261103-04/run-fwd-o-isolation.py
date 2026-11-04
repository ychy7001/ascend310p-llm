#!/usr/bin/env python3
"""候选Profile结束后，独占NPU重复调用固定输入的生产FwdO，结束恢复a2临时unit。"""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();native=root/'native-gdn'
out=root/'fwd-o-isolation';out.mkdir(exist_ok=False)
steps=[];paused=False;finished=False

def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
def run(label,cmd):
 state('running',current=label);start=time.monotonic()
 with (out/(label+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-start))
 if r.returncode:raise RuntimeError(label+' failed')
try:
 state('waiting_profile')
 while True:
  j=json.loads((root/'production-wy-batching-recheck/status.json').read_text())
  if j['state'] in ('failed','not_started'):state('not_started',reason='prior experiment failed');raise SystemExit(0)
  if j['state']=='complete':break
  time.sleep(5)
 paused=True
 run('pause-a2-temporary-unit',['sudo','-n','systemctl','stop','llm-backend-primary.service'])
 idle=0
 for _ in range(90):
  r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  idle=idle+1 if 'No running processes found' in r.stdout else 0
  if idle>=2:break
  time.sleep(2)
 else:raise RuntimeError('NPU not idle')
 cmd=json.loads((native/'validation-command.json').read_text())
 cmd[cmd.index('--name')+1]='qwen36-fwd-o-isolation-20261103'
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/tools
python3 probe_fwd_o_frozen.py --work /work --out-dir /work/fwd-o-isolation
'''
 run('fwd-o-frozen',cmd)
 result=json.loads((native/'fwd-o-isolation/summary.json').read_text())
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 finished=True
except Exception as e:
 state('failed',error=str(e));raise
finally:
 if paused:
  with (out/'restore-a2-service.log').open('x') as log:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit')
 if finished:state('complete',scope='fixed FwdO inputs, no model performance claim')
