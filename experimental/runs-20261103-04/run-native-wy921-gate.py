#!/usr/bin/env python3
"""等待无设备构建，独占a2跑WY门禁，finally恢复临时原服务并检查健康。"""
import argparse,json,subprocess,time,urllib.request,signal
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--attempt',default='v1');a=p.parse_args();assert a.attempt.isalnum();r=a.root.resolve();out=r/('native-wy921-gate-'+a.attempt);out.mkdir(exist_ok=False);paused=False;container=None
state_data={}
def state(value,**kw):state_data.update(state=value,**kw);(out/'status.json').write_text(json.dumps(state_data,indent=2)+'\n')
def health():
 try:
  with urllib.request.urlopen('http://127.0.0.1:18081/health',timeout=5) as h:return h.status==200
 except Exception:return False
def interrupted(n,f):raise RuntimeError('Interrupted signal '+str(n))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
 state('waiting_build')
 for _ in range(240):
  s=json.loads((r/'native-wy921-v1/status.json').read_text())
  if s['state']=='failed':raise RuntimeError('WY build failed; original service unchanged')
  if s['state']=='built_requires_numerical_gate':break
  time.sleep(5)
 else:raise RuntimeError('WY build wait timeout')
 assert health(),'Original service not healthy before exclusive test'
 state('pausing_experiment_service');paused=True;subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'],check=True,timeout=90)
 for _ in range(60):
  s=subprocess.check_output(['npu-smi','info'],text=True)
  if 'No running processes found' in s:break
  time.sleep(2)
 else:raise RuntimeError('NPU remains busy')
 template=json.loads((r/'upstream-gdn-pair921-v1/strict-device0-command.json').read_text());image='d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a';summaries=[]
 for device in (0,1):
  cmd=template.copy();container='wy921-gate-'+str(device)+'-'+str(time.time_ns());cmd[cmd.index('--name')+1]=container;idx=cmd.index(image);cmd[idx:idx]=['-v',str(r/'native-wy921-v1')+':/wy:rw','-v',str(out)+':/gate:rw'];cmd[-1]=f'''set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="/pair/opp/vendors/hyperqwen_gdn_pair921_transformer:/wy/opp/vendors/hyperqwen_wy921_transformer:${{ASCEND_CUSTOM_OPP_PATH:-}}"
export PYTHONPATH="/wy/tools:/pair/tools:${{PYTHONPATH:-}}"
python3 /wy/tools/check_native_wy921.py --pair /pair --wy /wy --device {device} --out /gate/device{device}.jsonl
'''
  (out/f'device{device}-command.json').write_text(json.dumps(cmd,indent=2));state('validating',device=device)
  with (out/f'device{device}.log').open('x') as f:res=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  if res.returncode:raise RuntimeError(f'Device{device} gate exit={res.returncode}')
  summaries.append(json.loads((out/f'device{device}.jsonl').read_text().splitlines()[-1]))
 state('passed_operator_gate',summaries=summaries,scope='Synthetic operator correctness only; real replay/model fixed-token gate still required before end-to-end performance')
except Exception as exc:state('failed',error=str(exc))
finally:
 if container:subprocess.run(['podman','stop','--time','5',container],capture_output=True,timeout=30)
 if paused:
  subprocess.run(['sudo','-n','systemctl','reset-failed','llm-backend-primary.service'],capture_output=True,timeout=30);res=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],capture_output=True,timeout=90)
  for _ in range(120):
   if health():break
   time.sleep(5)
  restoration={'start_returncode':res.returncode,'independent_health_200':health()};(out/'service-restoration.json').write_text(json.dumps(restoration,indent=2));state_data['restoration']=restoration
 else:state_data['original_service_unchanged_health_200']=health()
 state(state_data['state'])
