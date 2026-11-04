#!/usr/bin/env python3
"""在Dense启动诊断清理后逐进程运行合成量化Matmul；不读取模型权重。"""
import argparse
import json
from pathlib import Path
import subprocess
import time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();out=root/'dense-quant-isolation';out.mkdir(exist_ok=False)
steps=[];paused=False;finished=False

def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
try:
 state('waiting_previous_diagnostic_cleanup')
 while True:
  f=root/'qwen38-dense-baseline-sync-eager-diagnostic/status.json';j=json.loads(f.read_text()) if f.exists() else {}
  ready=j.get('state') in ['complete','failed'] and (f.parent/'restore-a2-service.log').exists()
  active=True if ready else False
  if ready and active:break
  time.sleep(5)
 paused=True;subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'],check=True)
 idle=0
 for _ in range(90):
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  idle=idle+1 if 'No running processes found' in info.stdout else 0
  if idle>=2:break
  time.sleep(2)
 else:raise RuntimeError('NPU not idle')
 template=json.loads((root/'native-gdn/validation-command.json').read_text());image=json.loads((root/'production-manifest.json').read_text())['image']
 # Each potentially failing device call has its own process/container/context.
 cases=[(1,5120,17408,'production'),(4096,5120,17408,'production'),(4096,5120,17408,'transpose-then-nz'),(4096,5120,17408,'nd'),(4096,8704,5120,'production'),(1,8704,5120,'production'),(4096,8704,5120,'transpose-then-nz'),(4096,8704,5120,'nd')]
 for i,(m,k,n,layout) in enumerate(cases):
  label=f'{i}-m{m}-k{k}-n{n}-{layout}';state('running',current=label)
  cmd=template.copy();cmd[cmd.index('--name')+1]='dense-quant-isolation-'+label+'-'+str(time.time_ns())
  idx=cmd.index(image);cmd[idx:idx]=['-v',str(out)+':/quantdiag:rw']
  cmd[-1]=f'''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_LAUNCH_BLOCKING=1
python3 /work/tools/probe_dense_quant_matmul.py --m {m} --k {k} --n {n} --layout {layout} --out /quantdiag/{label}.json
'''
  (out/(label+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n');start=time.monotonic()
  with (out/(label+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-start))
 finished=True
except Exception as error:state('failed',error=str(error));raise
finally:
 if paused:
  with (out/'restore-a2-service.log').open('x') as log:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit')
 if finished:state('complete',scope='Diagnostic execution complete; inspect each case status, not a correctness acceptance')
