"""Independent CPU triangular reference for the new candidate; original tolerances."""
import hashlib,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'/kernel')
from wy_snapshot_reference import reference,metrics,OUTPUT_NAMES
r=Path('/work');old=r/'cpu-old';boundary=r/'cpu-boundary';device=r/'neumaier-g0';out=r/'neumaier-cpu';out.mkdir(exist_ok=False)
torch.set_num_threads(4);start=time.monotonic();rows=[]
selection=json.loads((r/'boundary-analysis.json').read_text())['rows'];selection=[x for x in selection if x['arm']=='B' and x['stage']=='WY'];assert len(selection)==480
sets=[('snapshots',old/'wy921-localize-20261104/raw/captures'),('triggers',old/'wy921-fix-20261105/raw/triggers')]
items=[]
for label,source in sets:
 for path in sorted(source.glob('rank*-call*.pt')):items.append((label,path,device/'wy'/label/path.name,None))
assert len(items)==132
items += [('boundaries',boundary/row['file'],device/'boundaries'/f'{i:04d}.pt',row['sha256']) for i,row in enumerate(selection)]
totals={label:{mode:{n:{'squared_error':0.,'outside':0} for n in OUTPUT_NAMES} for mode in ('native','torch')} for label in ('snapshots','triggers','boundaries')}
for i,(label,path,replay,expected_sha) in enumerate(items):
 if expected_sha:assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha
 saved=torch.load(path,map_location='cpu',weights_only=False);data=torch.load(replay,map_location='cpu',weights_only=True);expected=reference(saved['inputs'])
 row={'set':label,'source':str(path.name),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 for mode in ('native','torch'):
  actual=data[mode+'_outputs'];row[mode+'_vs_cpu']=metrics(actual,expected)
  for n,a,e,m in zip(OUTPUT_NAMES,actual,expected,row[mode+'_vs_cpu']):
   totals[label][mode][n]['squared_error']+=float((a.float()-e).double().square().sum())
   totals[label][mode][n]['outside']+=m['outside_tolerance']
 rows.append(row)
 with (out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
 if (i+1)%30==0:print(i+1,'/',len(items),flush=True)
repeat_old=[json.loads(s) for s in (device/'wy/replay.jsonl').read_text().splitlines()]
repeat_boundary=[json.loads(s) for s in (device/'boundaries/replay.jsonl').read_text().splitlines()]
queued=json.loads((device/'queued-repeat.json').read_text());assert len(repeat_old)==132 and len(repeat_boundary)==480
synth=[]
for rank in (0,1):
 values=[json.loads(s) for s in (device/f'synthetic-d{rank}.jsonl').read_text().splitlines()]
 summary=next(v for v in values if v['kind']=='summary');assert summary['cases']==15
 synth.append(summary)
by_set={label:{'cases':sum(r['set']==label for r in rows),'native_passed':sum(r['set']==label and all(m['passed'] for m in r['native_vs_cpu']) for r in rows),
 'torch_passed':sum(r['set']==label and all(m['passed'] for m in r['torch_vs_cpu']) for r in rows),
 'no_worse_control':{n:v['native'][n]['squared_error']<=v['torch'][n]['squared_error']+1e-12 and v['native'][n]['outside']<=v['torch'][n]['outside'] for n in OUTPUT_NAMES}} for label,v in totals.items()}
summary={'scope':'Raw-K library Gram + FP32 beta postscale + Neumaier scan vs frozen independent CPU FP32 reference; no model/performance gate',
 'wy_cases':len(rows),'sets':by_set,'totals':totals,'synthetic':synth,
 'serial_repeat_passed':all(x['repeat_pass'] for x in repeat_old+repeat_boundary),
 'queued_repeat_passed':queued['passed'] and all(x['repeat_pass'] for x in repeat_boundary),'queued_legacy_cases':queued['cases'],
 'candidate_manifest_sha256':json.loads((device/'freeze.json').read_text())['wy_manifest_sha256'],'elapsed_s':time.monotonic()-start}
summary['passed']=len(rows)==612 and all(v['native_passed']==v['cases'] and all(v['no_worse_control'].values()) for v in by_set.values()) and all(s['passed']==s['cases'] for s in synth) and summary['serial_repeat_passed'] and summary['queued_repeat_passed']
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
raise SystemExit(0 if summary['passed'] else 1)
