#!/usr/bin/env python3
"""同镜像原始GDN路径，独占对照CANN默认executor cache与ACLNN_CACHE_LIMIT=0。"""
import argparse
import json
from pathlib import Path
import subprocess
import time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();out=root/'executor-cache-control';out.mkdir(exist_ok=False)
steps=[];paused=False;finished=False

def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
try:
 state('checking_previous_cleanup')
 if json.loads((root/'dense-quant-captured-replay-v2/status.json').read_text())['state']!='complete':raise RuntimeError('Previous diagnostic incomplete')
 paused=True;subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'],check=True)
 idle=0
 for _ in range(90):
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  idle=idle+1 if 'No running processes found' in info.stdout else 0
  if idle>=2:break
  time.sleep(2)
 else:raise RuntimeError('NPU not idle')
 template=json.loads((root/'native-gdn/validation-command.json').read_text());image=json.loads((root/'production-manifest.json').read_text())['image']
 for label,setting in [('default','unset ACLNN_CACHE_LIMIT'),('disabled','export ACLNN_CACHE_LIMIT=0')]:
  cmd=template.copy();cmd[cmd.index('--name')+1]='executor-cache-'+label+'-'+str(time.time_ns())
  idx=cmd.index(image);cmd[idx:idx]=['-v',str(out)+':/cache-probe:rw']
  cmd[-1]=f'''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
{setting}
cd /work/tools
python3 repro_chunk_gdn_nondeterminism.py --out /cache-probe/{label}-repro.jsonl --repeats 48
python3 probe_gdn_intermediates.py --work /work --out-dir /cache-probe/{label}-chain
'''
  (out/(label+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n');state('running',current=label);begin=time.monotonic()
  with (out/(label+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-begin))
  if r.returncode:raise RuntimeError(label+' failed')
 finished=True
except Exception as error:state('failed',error=str(error));raise
finally:
 if paused:
  with (out/'restore-a2-service.log').open('x') as log:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit')
 if finished:state('complete',scope='Diagnostic cache control only; inspect raw metrics before any conclusion')
