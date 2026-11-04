#!/usr/bin/env python3
"""Final cleanup/independent health check; never starts an experiment workload."""
import datetime,json,pathlib,subprocess
out=pathlib.Path('results/decode-attribution-20261104/v2')
remote=r'''
import datetime,json,subprocess,time,urllib.request
r={"time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
 p=subprocess.run(["podman","stop","--time","5","decode-2-export-experiment"],capture_output=True,text=True,timeout=20)
 r['export_stop_returncode']=p.returncode
except subprocess.TimeoutExpired:r['export_stop_timeout']=True
active=subprocess.run(["sudo","-n","systemctl","is-active","llm-backend-primary.service"],capture_output=True,text=True).stdout.strip()
if active!='active':
 subprocess.run(["sudo","-n","systemctl","start","llm-backend-primary.service"],capture_output=True,timeout=120)
for _ in range(90):
 try:
  r['health']=urllib.request.urlopen('http://127.0.0.1:18081/health',timeout=3).status
  if r['health']==200:break
 except Exception:r['health']=None
 time.sleep(2)
r['unit']=subprocess.run(["sudo","-n","systemctl","is-active","llm-backend-primary.service"],capture_output=True,text=True).stdout.strip()
r['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(r))
'''
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=subprocess.run(['ssh','-o','ConnectTimeout=8','-o','ConnectionAttempts=1','ychy-a2','python3 -'],input=remote,text=True,capture_output=True,timeout=360)
(out/'raw/final-recovery-ssh.txt').write_text(p.stdout+p.stderr)
latest={'attempt_utc':started,'ssh_returncode':p.returncode,'independent_health_status':None,'connected':p.returncode!=255}
if p.returncode==0:
 d=json.loads(p.stdout.splitlines()[-1]);latest.update(d,independent_health_status=d.get('health'))
result={'last_successful_independent_confirmation':{'utc':'2026-11-04T09:38:55.974152+00:00','health':200,'unit':'active','experimental_inference_container_running':False,'source':'independent SSH check after controller finally restoration, before offline export'},'latest_confirmation_attempt':latest,'failure':'SSH and Tailscale ping became unreachable during CPU offline export; temporal association only, cause not established','current_primary_service_status':'confirmed' if latest['independent_health_status']==200 else 'unknown after connectivity loss','cpu_export_container_final_status':'stop attempted on final SSH connection' if latest['connected'] else 'unknown; final cleanup command could not connect','new_npu_experiments_after_connectivity_loss':0,'new_four_window_profile_exports_retrieved':False,'stopped_extra_work':True,'half_day_deadline_utc':'2026-11-04T14:23:32Z','observation_issues':[{'attempt':1,'issue':'optional reset_prefix_cache route returned404; no generation; finally restored200'},{'attempt':2,'issue':'mapping dispatch mode lacked higher-order-operator support; cold replay and B1/B2 had already completed; finally restored200'},{'attempt':3,'issue':'mapping succeeded3forwards/rank, but Level2 B3 profiling caused severe overhead and only63stable steps; excluded from performance; finally restored200'},{'attempt':4,'issue':'four Level1 lean windows completed; finally restored200 and independent confirmation succeeded; subsequent offline-export SSH connection lost'}]}
(out/'recovery-and-limitations.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
