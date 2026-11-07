"""Independent frozen CPU oracle adjudication for Torch blocked-UT real inputs."""
import hashlib,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'/kernel')
from wy_snapshot_reference import reference,metrics,OUTPUT_NAMES
r=Path('/work'); device=r/'torch-ut-g0-20261107';out=r/'torch-ut-cpu-20261107';out.mkdir(exist_ok=False)
torch.set_num_threads(4);start=time.monotonic()
freeze=json.loads((device/'freeze.json').read_text()); records=[json.loads(s) for s in (device/'rows.jsonl').read_text().splitlines()];assert len(records)==len(freeze['inputs'])==612
summary={'scope':'612 real-input G0 subset; no synthetic H/O or model acceptance','sets':{},'totals':{}};all_pass=True
for i,(item,rec) in enumerate(zip(freeze['inputs'],records)):
 label=item['set']; rel=Path(item['file'])
 if label=='snapshots': path=r/'cpu-old/wy921-localize-20261104/raw/captures'/rel.name
 elif label=='triggers':path=r/'cpu-old/wy921-fix-20261105/raw/triggers'/rel.name
 else:path=r/'cpu-boundary'/Path(*rel.parts[2:])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha'],path
 saved=torch.load(path,map_location='cpu',weights_only=False);data=torch.load(device/f'{i:04d}.pt',weights_only=True);expected=reference(saved['inputs'])
 totals=summary['totals'].setdefault(label,{mode:{n:{'squared_error':0.,'outside':0} for n in OUTPUT_NAMES} for mode in ('candidate','baseline')})
 counts=summary['sets'].setdefault(label,{'cases':0,'candidate_passed':0,'baseline_passed':0,'repeat_passed':0,'qkg_exact':0});counts['cases']+=1
 row={'index':i,'set':label}
 for mode,key in [('candidate','native_outputs'),('baseline','torch_outputs')]:
  actual=data[key]; row[mode]=metrics(actual,expected)
  counts[mode+'_passed']+=int(all(v['passed'] for v in row[mode]))
  for n,a,e,m in zip(OUTPUT_NAMES,actual,expected,row[mode]):
   totals[mode][n]['squared_error']+=float((a.float()-e).double().square().sum());totals[mode][n]['outside']+=m['outside_tolerance']
 counts['repeat_passed']+=int(rec['repeat_pass']);counts['qkg_exact']+=int(rec['qkg_exact'])
 with (out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
for label,totals in summary['totals'].items():
 counts=summary['sets'][label];counts['no_worse']={n:totals['candidate'][n]['squared_error']<=totals['baseline'][n]['squared_error']+1e-12 and totals['candidate'][n]['outside']<=totals['baseline'][n]['outside'] for n in OUTPUT_NAMES}
 all_pass &= counts['candidate_passed']==counts['cases'] and counts['repeat_passed']==counts['cases'] and counts['qkg_exact']==counts['cases'] and all(counts['no_worse'].values())
summary['passed']=bool(all_pass);summary['elapsed_s']=time.monotonic()-start
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
raise SystemExit(0 if all_pass else 1)
