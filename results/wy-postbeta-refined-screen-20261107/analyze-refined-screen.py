"""Independently audit raw HTTP choices, controls, and loaded native libraries."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=a.record;screen=r/'screen';policy=read(r/'selected-policy.json');freeze=read(screen/'freeze.json');closed=read(screen/'window-closed.json');g0=read(r/'cpu-adjudication.json')
assert freeze['policy_sha']==sha(r/'selected-policy.json') and freeze['g0_sha']==sha(r/'cpu-adjudication.json')
assert g0['passed'] and closed['cleanup_passed'] and closed['error'] is None
assert read(screen/'status.json')['state']=='complete'
rows=[];identities={};native_calls=[]
for arm in ('A','A_prime','B'):
 record=read(screen/f'screen-{arm}-results.json');assert len(record['points'])==5
 inspect=read(screen/f'screen-{arm}-container-inspect.json')[0];state=inspect['State'];assert state['ExitCode']==0 and not state['OOMKilled'] and not state['Running']
 command=read(screen/f'screen-{arm}-launch-command.json');assert command[command.index('--name')+1]==inspect['Name'].lstrip('/')
 log=(screen/f'screen-{arm}-server.log').read_text();workers=[]
 for line in log.splitlines():
  if '[wy_gate_worker] ' in line:
   body=line.split('[wy_gate_worker] ',1)[1];body=re.sub(r'\x1b\[[0-9;]*m','',body);workers.append(json.loads(body))
  if arm=='B' and '[wy_gate_native_call] ' in line:
   body=line.split('[wy_gate_native_call] ',1)[1];body=re.sub(r'\x1b\[[0-9;]*m','',body);call=json.loads(body);prefix=re.search(r'Worker_TP([01]) pid=(\d+)',line);assert prefix is not None;call['log_tp_rank']=int(prefix.group(1));call['log_pid']=int(prefix.group(2));native_calls.append(call)
 assert len(workers)==2 and all(v['arm']==('A' if arm=='A_prime' else arm) and v['fallback'] is False and v['wy_artifact_manifest_sha256']==g0['candidate_manifest_sha256'] for v in workers)
 identities[arm]={'container_exitcode':state['ExitCode'],'worker_records':workers,'server_log_sha':sha(screen/f'screen-{arm}-server.log')}
 for point in policy['points']:
  raw=read(screen/f'screen-{arm}-raw'/(point['case']+f"-p{point['position']:02d}.json"));choice=raw['choices'][0];top1=choice['token_ids'][0]
  saved=next(x for x in record['points'] if (x['case'],x['position'])==(point['case'],point['position']));assert saved['top1']==top1 and saved['context_sha256']==point['context_sha256']
  rows.append({'arm':arm,'case':point['case'],'position':point['position'],'classification':point['classification'],'top1':top1,'R_top1':point['R_top1'],'matches_R':top1==point['R_top1'],'raw_sha':sha(screen/f'screen-{arm}-raw'/(point['case']+f"-p{point['position']:02d}.json"))})
assert {v['log_tp_rank'] for v in native_calls}=={0,1}
expected='/native-wy/opp/vendors/hyperqwen_postbeta_refined_transformer/op_api/lib/libcust_opapi.so'
assert all(expected in v['loaded_apis'] for v in native_calls)
struct={arm:all(v['matches_R'] for v in rows if v['arm']==arm and v['classification']=='structural') for arm in ('A','A_prime','B')};eligible=all(struct.values());original=read(screen/'decision.json');assert original['eligible_for_full_G1']==eligible
result={'scope':'independent fixed-screen audit, not G1/G2/performance/adoption','rows':rows,'structural_match':struct,'controls_valid':struct['A'] and struct['A_prime'],'candidate_structural_match':struct['B'],'eligible_for_full_G1':eligible,'native_call_records':len(native_calls),'native_ranks':[0,1],'rank_identity_source':'Worker_TP rank and PID prefix; frozen worker LOCAL_RANK field is null','expected_library_verified':True,'identities':identities,'cleanup_passed':closed['cleanup_passed'],'elapsed_s':closed['elapsed_s'],'freeze_sha':sha(screen/'freeze.json'),'policy_sha':sha(r/'selected-policy.json'),'original_decision_sha':sha(screen/'decision.json'),'historical_fail_preserved':True}
with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','identities')},indent=2))
