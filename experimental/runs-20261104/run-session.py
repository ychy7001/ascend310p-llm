#!/usr/bin/env python3
"""One-off NPU session. Always restores the original system service."""
import argparse,datetime as dt,json,pathlib,signal,subprocess,sys,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();root=a.root.resolve()
deadline=dt.datetime(2026,11,4,14,23,32,tzinfo=dt.timezone.utc).timestamp()
log=root/'session-events.jsonl'
def emit(kind,**kw):
 with log.open('a') as f:f.write(json.dumps({'kind':kind,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**kw})+'\n')
 print(kind,kw,flush=True)
def health(port):
 try:return urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3).status
 except Exception:return None
def terminate(signum,frame):raise TimeoutError('session signal or half-day deadline')
for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(signum,terminate)
signal.alarm(max(1,int(deadline-time.time()-600)))
stopped=False
try:
 assert health(18081)==200,'Original service must be healthy'
 emit('original_healthy',status=200)
 stopped=True
 subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'],check=True,timeout=180)
 emit('production_stopped')
 subprocess.run([sys.executable,str(root/'launch-observed.py'),'--root',str(root)],check=True,timeout=120)
 for _ in range(300):
  if health(18106)==200:break
  if time.time()>deadline-900:raise TimeoutError('Insufficient time for experiment and restoration')
  state=subprocess.check_output(['podman','inspect','--format','{{.State.Running}}','decode-2-observed-20261104'],text=True).strip()
  if state!='true':raise RuntimeError('Observed replica stopped during startup')
  time.sleep(2)
 else:raise TimeoutError('Replica startup >600 sec')
 emit('replica_ready')
 workload_command=[sys.executable,str(root/'workloads.py'),'--root',str(root)]
 if (root/'profiles.only').exists():workload_command+=['--profiles-only']
 elif (root/'remaining.only').exists():workload_command+=['--only-remaining']
 subprocess.run(workload_command,check=True,timeout=min(6000,max(1,deadline-time.time()-900)))
 emit('workloads_completed')
except BaseException as exc:
 emit('session_error',type=type(exc).__name__,message=str(exc));raise
finally:
 signal.alarm(0)
 try:
  with (root/'replica.log').open('w') as f:subprocess.run(['podman','logs','--timestamps','decode-2-observed-20261104'],stdout=f,stderr=subprocess.STDOUT,timeout=60)
  subprocess.run(['podman','stop','--time','30','decode-2-observed-20261104'],timeout=75,check=False)
 except Exception as exc:emit('replica_cleanup_error',message=str(exc))
 if stopped:
  subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],check=False,timeout=180)
  for _ in range(180):
   if health(18081)==200:break
   time.sleep(2)
  emit('restoration',health=health(18081),active=subprocess.run(['systemctl','is-active','llm-backend-primary.service'],capture_output=True,text=True).stdout.strip())
