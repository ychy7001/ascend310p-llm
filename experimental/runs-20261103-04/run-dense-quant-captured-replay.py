#!/usr/bin/env python3
"""等待实际输入捕获轮清理后，逐进程重放；CPU采样校验，不修改生产权重。"""
import argparse
import json
import shutil
import hashlib
from pathlib import Path
import subprocess
import time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--attempt',default='');p.add_argument('--allocator',choices=['expandable','legacy','compare'],help='explicit allocator control before importing torch')
p.add_argument('--synthetic',action='store_true',help='generated zero activation/int8 weight; no checkpoint reads')
p.add_argument('--focused-pair',action='store_true',help='rank1/device0 captured full rows only')
p.add_argument('--repeats',type=int,default=1)
p.add_argument('--rank-device-grid',action='store_true',help='cross both captured TP ranks on both physical devices');a=p.parse_args()
if a.repeats<1:p.error('repeats must be positive')
root=a.root.resolve();out=root/('dense-quant-captured-replay'+('-'+a.attempt if a.attempt else ''));out.mkdir(exist_ok=False)
shutil.copy2(__file__,out/'supervisor-used.py')
shutil.copy2(root/'native-gdn/tools/replay_dense_quant_capture.py',out/'operator-probe.py')
(out/'probe-freeze.json').write_text(json.dumps({'sha256':hashlib.sha256((out/'operator-probe.py').read_bytes()).hexdigest()})+'\n')
steps=[];paused=False;finished=False

def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
try:
 if a.attempt:
  state('waiting_previous_replay_cleanup')
  while json.loads((root/'dense-quant-captured-replay/status.json').read_text()).get('state')!='complete':time.sleep(5)
 state('waiting_capture_cleanup');capture=root/'qwen38-dense-baseline-quant-trace'
 while True:
  f=capture/'status.json';j=json.loads(f.read_text()) if f.exists() else {}
  if j.get('state') in ['complete','failed'] and (capture/'restore-a2-service.log').exists():break
  time.sleep(5)
 if a.rank_device_grid and not (capture/'quant-matmul-case-rank1.pt').exists():raise RuntimeError('Rank1 capture missing')
 if not (capture/'quant-matmul-case-rank0.pt').exists():raise RuntimeError('No actual down_proj capture; no replay')
 paused=True;subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'],check=True)
 idle=0
 for _ in range(90):
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  idle=idle+1 if 'No running processes found' in info.stdout else 0
  if idle>=2:break
  time.sleep(2)
 else:raise RuntimeError('NPU not idle')
 template=json.loads((root/'native-gdn/validation-command.json').read_text());image=json.loads((root/'production-manifest.json').read_text())['image']
 cases=[('captured',None,rank,device) for rank in (0,1) for device in (0,1)] if a.rank_device_grid else [(layout,rows,0,0) for layout,rows in [('captured',None),('nd',None),('nz',None),('captured',1),('captured',4),('captured',64)]]
 if a.synthetic:cases=[('captured',None,0,0)]
 if a.focused_pair:cases=[('captured',None,1,0)]
 allocators=['legacy','expandable'] if a.allocator=='compare' else [a.allocator]
 cases=[(*case,allocator,repeat) for repeat in range(a.repeats) for case in cases for allocator in allocators]
 for i,(layout,rows,rank,device,allocator,repeat) in enumerate(cases):
  label=f'{i}-rank{rank}-device{device}-{layout}-rows{rows or "all"}-{allocator or "default"}-r{repeat}';state('running',current=label)
  cmd=template.copy();cmd[cmd.index('--name')+1]='dense-quant-captured-'+label+'-'+str(time.time_ns())
  idx=cmd.index(image);cmd[idx:idx]=['-v',str(out)+':/quantreplay:rw','-v',str(capture)+':/capture:ro']
  limit=f' --rows {rows}' if rows else ''
  case_arg=f'--synthetic --seed {310+repeat}' if a.synthetic else f'--case /capture/quant-matmul-case-rank{rank}.pt'
  allocator=f'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:{"True" if allocator=="expandable" else "False"}' if allocator else ''
  cmd[-1]=f'''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_LAUNCH_BLOCKING=1
{allocator}
python3 /quantreplay/operator-probe.py {case_arg} --device {device} --activation-layout {layout} --out /quantreplay/{label}.json{limit}
'''
  (out/(label+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n');begin=time.monotonic()
  try:
   with (out/(label+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  except subprocess.TimeoutExpired:
   subprocess.run(['podman','stop','--time','5',cmd[cmd.index('--name')+1]],capture_output=True,text=True,timeout=30)
   steps.append(dict(label=label,returncode=None,timed_out=True,elapsed_s=time.monotonic()-begin))
   raise TimeoutError('Replay exceeded 180s; inspect result marker to distinguish initialization from operator execution')
  steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-begin))
  if not (out/(label+'.json')).exists():raise RuntimeError('Replay did not reach tensor preparation; inspect bootstrap/serialization error before further cases')
 finished=True
except Exception as error:state('failed',error=str(error));raise
finally:
 if paused:
  with (out/'restore-a2-service.log').open('x') as log:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit')
 if finished:state('complete',scope='Captured replay diagnostics complete; per-case correctness is reported separately')
