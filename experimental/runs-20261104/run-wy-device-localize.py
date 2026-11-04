#!/usr/bin/env python3
"""仅设备算子阶段；承接已恢复的观测阶段，独占设备并 finally 恢复服务。"""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import time

spec=importlib.util.spec_from_file_location('model_gate',Path(__file__).with_name('run-wy-model-gate.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--tools',type=Path,required=True)
p.add_argument('--out-name',required=True)
p.add_argument('--candidate',choices=('native','cann'),required=True)
p.add_argument('--cann',type=Path)
p.add_argument('--captures',type=Path,required=True)
p.add_argument('--full-gdn',action='store_true')
p.add_argument('--synthetic',action='store_true')
a=p.parse_args()
x=m.Gate.__new__(m.Gate)
x.root=a.root;x.tools=a.tools;x.out=a.root/a.out_name;x.out.mkdir()
x.container=None;x.paused=False;x.steps=[];x.started=time.monotonic();x.mode=None;x.error=None
signal.signal(signal.SIGTERM,m.interrupted);signal.signal(signal.SIGINT,m.interrupted)
try:
 assert x.health('http://127.0.0.1:18081')
 assert subprocess.run(['systemctl','--user','is-active','qwen36-image.service'],capture_output=True,text=True).stdout.strip()=='inactive'
 assert not subprocess.check_output(['podman','ps','-q'],text=True).strip()
 x.paused=True
 subprocess.run(['sudo','-n','systemctl','stop',m.SERVICE],check=True)
 idle=subprocess.check_output(['npu-smi','info'],text=True)
 (x.out/'idle-before.txt').write_text(idle);assert 'No running processes found' in idle
 x.container='wy921-device-'+a.candidate+'-'+str(time.time_ns())
 command=['podman','run','-d','--pull=never','--name',x.container,'--network','host',
          '--group-add','keep-groups','--ipc','shareable','--shm-size','32g','--entrypoint','/bin/bash']
 for device in ('davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc'):command+=['--device','/dev/'+device]
 mounts=[(a.root,'/runtime','rw'),(a.tools,'/tools','ro'),(a.root/'upstream-gdn-pair921-v1','/gdn-pair','ro'),(a.root/'native-wy921-v1','/native-wy','ro')]
 for name in ('/usr/local/bin/npu-smi','/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/Ascend/driver/lib64','/usr/local/dcmi'):mounts.append((name,name,'ro'))
 if a.cann:mounts.append((a.cann,'/cann-wy','ro'))
 for src,dst,mode in mounts:command+=['-v',f'{src}:{dst}:{mode}']
 script='set -e; source /usr/local/Ascend/ascend-toolkit/set_env.sh; export PYTHONPATH=/tools:/gdn-pair/tools:${PYTHONPATH:-}; export OMP_NUM_THREADS=1; exec python3 /tools/replay-wy-snapshots.py --pair /gdn-pair --wy /native-wy --captures /runtime/'+str(a.captures.relative_to(a.root))+' --out /runtime/'+a.out_name+'/replayed --candidate '+a.candidate
 if a.cann:script+=' --cann /cann-wy'
 if a.full_gdn:
  wyroot='/native-wy' if a.candidate=='native' else '/cann-wy'
  script='set -e; source /usr/local/Ascend/ascend-toolkit/set_env.sh; export PYTHONPATH=/tools:/gdn-pair/tools:${PYTHONPATH:-}; export OMP_NUM_THREADS=1; exec python3 /tools/replay-real-gdn.py --pair /gdn-pair --wy '+wyroot+' --captures /runtime/'+str(a.captures.relative_to(a.root))+' --out /runtime/'+a.out_name+'/full-gdn --candidate '+a.candidate
 if a.synthetic:
  assert a.candidate=='cann' and a.cann
  prefix='set -e; source /usr/local/Ascend/ascend-toolkit/set_env.sh; export PYTHONPATH=/tools:/gdn-pair/tools:${PYTHONPATH:-}; export OMP_NUM_THREADS=2; '
  tests=''
  for device in (0,1):
   tests+='python3 /tools/check-cann-wy921.py --pair /gdn-pair --wy /cann-wy --device '+str(device)+' --out /runtime/'+a.out_name+'/synthetic-d'+str(device)+'.jsonl; '
  script=prefix+tests+script
 command+=[m.IMAGE,'-lc',script]
 x.run('device-launch',command)
 code=subprocess.check_output(['podman','wait',x.container],text=True).strip()
 x.stop('device')
 if code!='0':raise RuntimeError('Device replay exited '+code)
except Exception as error:
 x.error=repr(error);x.state('failed',error=x.error)
finally:
 x.finish()
 image=subprocess.run(['systemctl','--user','is-active','qwen36-image.service'],capture_output=True,text=True).stdout.strip()
 m.write(x.out/'image-service-final.json',{'state':image,'started_by_experiment':False})
 assert image=='inactive'
if x.error:raise RuntimeError(x.error)
