import json, subprocess, fcntl, time, hashlib
from pathlib import Path
base=Path(__file__).resolve().parent
out=base/'torch-ut-scripted-identity-window-20261107';out.mkdir(exist_ok=False)
name='wy-torch-ut-scripted-identity-20261107'
started=time.time(); code=None; error=None
with open('/tmp/a310-npu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['podman','ps','-q'],text=True).strip()==''
 assert subprocess.check_output(['sudo','-n','podman','ps','-q'],text=True).strip()==''
 assert all(subprocess.run(c,capture_output=True,text=True).stdout.strip()=='inactive' for c in [['systemctl','is-active','llm-backend-primary.service'],['systemctl','--user','is-active','qwen36-image.service']])
 command=json.loads((base/'wy4-g0-v3/launch-command.json').read_text())
 command[command.index('--name')+1]=name
 command[-1]='source /usr/local/Ascend/cann/set_env.sh; export ASCEND_CUSTOM_OPP_PATH=/runtime/upstream-gdn-pair921-v1/opp/vendors/hyperqwen_gdn_pair921_transformer; export OMP_NUM_THREADS=1; python3 /runtime/replay-torch-ut-scripted-identity.py'
 (out/'launch.json').write_text(json.dumps(command,indent=2))
 try:
  subprocess.run(command,check=True,capture_output=True,timeout=30)
  code=subprocess.check_output(['podman','wait',name],text=True,timeout=900).strip()
 except Exception as exc: error=repr(exc)
 finally:
  subprocess.run(['podman','stop','--time','5',name],capture_output=True,timeout=20)
  with (out/'device.log').open('w') as log: subprocess.run(['podman','logs',name],stdout=log,stderr=subprocess.STDOUT,timeout=20)
  result={ 'elapsed_s':time.time()-started,'container_exit':code,'error':error,
   'running_rootless':subprocess.check_output(['podman','ps','-q'],text=True).strip(),
   'running_rootful':subprocess.check_output(['sudo','-n','podman','ps','-q'],text=True).strip(),
   'npu':subprocess.check_output(['npu-smi','info'],text=True),
   'services':{k:subprocess.run(v,capture_output=True,text=True).stdout.strip() for k,v in {
    'primary':['systemctl','is-active','llm-backend-primary.service'],
    'image':['systemctl','--user','is-active','qwen36-image.service']}.items()}}
  result['cleanup_passed']=not result['running_rootless'] and not result['running_rootful'] and 'No running processes' in result['npu'] and all(v=='inactive' for v in result['services'].values())
  (out/'closed.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
