"""Bounded isolated UT graph pilot, shared device lock, independent cleanup."""
import argparse,fcntl,hashlib,json,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();base=a.root;out=base/'torch-wy-fullgraph-20261107';out.mkdir(exist_ok=False)
def write(name,v):
 with (out/name).open('x') as f:json.dump(v,f,indent=2)
def snapshot():
 cmds={'rootless':['podman','ps','-q'],'rootful':['sudo','-n','podman','ps','-q'],'primary':['systemctl','is-active','llm-backend-primary.service'],'image':['systemctl','--user','is-active','qwen36-image.service'],'npu':['npu-smi','info']}
 return {k:{'code':v.returncode,'stdout':v.stdout.strip()} for k,c in cmds.items() for v in [subprocess.run(c,capture_output=True,text=True,timeout=30)]}
name='wy-torch-ut-graph-'+str(time.time_ns());error=None;code=None;start=None
with open('/tmp/a310-npu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();start_ns=time.time_ns()
 def interrupted(*_):raise TimeoutError('UT graph pilot action budget')
 signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted);signal.alarm(900)
 try:
  before=snapshot();write('window-start.json',before);assert before['rootless']['stdout']==before['rootful']['stdout']=='';assert before['primary']['stdout']==before['image']['stdout']=='inactive'
  write('freeze.json',{'controller_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'probe_sha':hashlib.sha256((base/'probe-torch-wy-fullgraph.py').read_bytes()).hexdigest(),'baseline_sha':hashlib.sha256((base/'runtime-wy-pinned.py').read_bytes()).hexdigest(),'action_budget_s':900,'scope':'two devices; full original Torch WY graph capture, all input copies/output clones counted;640 real/3840 synthetic; no model/performance acceptance'})
  command=json.loads((base/'wy4-g0-v3/launch-command.json').read_text());command[command.index('--name')+1]=name;command[-1]='set -e; source /usr/local/Ascend/cann/set_env.sh; export ASCEND_CUSTOM_OPP_PATH=/runtime/upstream-gdn-pair921-v1/opp/vendors/hyperqwen_gdn_pair921_transformer; export OMP_NUM_THREADS=1; python3 /runtime/probe-torch-wy-fullgraph.py';write('launch-command.json',command)
  subprocess.run(command,check=True,capture_output=True,timeout=60);code=subprocess.check_output(['podman','wait',name],text=True,timeout=850).strip();assert code=='0',code
 except BaseException as exc:error=repr(exc)
 finally:
  signal.alarm(0);detail=subprocess.run(['podman','inspect',name],capture_output=True,text=True,timeout=30)
  if detail.returncode==0:
   inspected=json.loads(detail.stdout)
   if inspected[0]['State']['Running']:subprocess.run(['podman','stop','--time','30',name],check=True,capture_output=True,timeout=60);inspected=json.loads(subprocess.check_output(['podman','inspect',name],text=True))
   write('container-inspect.json',inspected)
   with (out/'device.log').open('x') as f:subprocess.run(['podman','logs',name],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
  after=snapshot();clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and after['primary']['stdout']==after['image']['stdout']=='inactive' and 'No running processes' in after['npu']['stdout'];write('window-closed.json',{'start_unix_ns':start_ns,'end_unix_ns':time.time_ns(),'elapsed_s':time.monotonic()-start,'container_exitcode':code,'error':error,'cleanup_passed':clean,'snapshot':after});assert clean
if error:raise RuntimeError(error)
