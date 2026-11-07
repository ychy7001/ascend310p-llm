"""Read-only scan rounding diagnosis; no oracle or kernel changes."""
import hashlib,json,time
from pathlib import Path
import torch
r=Path('/work');out=r/'wy-scan-rounding-20261107';out.mkdir(exist_ok=False)
torch.set_num_threads(4);start=time.monotonic()
freeze=json.loads((r/'torch-ut-g0-20261107/freeze.json').read_text())
summary={'scope':'CPU emulation of FP32 sequential/compensated scans, not device correctness','cases':0,'modes':{}}
for i,item in enumerate(freeze['inputs']):
 rel=Path(item['file']);label=item['set']
 if label=='snapshots':path=r/'cpu-old/wy921-localize-20261104/raw/captures'/rel.name
 elif label=='triggers':path=r/'cpu-old/wy921-fix-20261105/raw/triggers'/rel.name
 else:path=r/'cpu-boundary'/Path(*rel.parts[2:])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha']
 saved=torch.load(path,map_location='cpu',weights_only=False)
 gate=saved['inputs'][3].float().transpose(1,2);b,h,t=gate.shape;gate=gate.reshape(b,h,-1,64)
 expected=gate.cumsum(-1)
 naive=torch.zeros_like(gate[...,0]);ks=naive.clone();kc=naive.clone();ns=naive.clone();nc=naive.clone()
 values={'sequential':[],'kahan':[],'neumaier':[]}
 for n in range(64):
  x=gate[...,n];naive=naive+x;values['sequential'].append(naive)
  y=x-kc;tmp=ks+y;kc=(tmp-ks)-y;ks=tmp;values['kahan'].append(ks)
  tmp=ns+x;nc=nc+torch.where(ns.abs()>=x.abs(),(ns-tmp)+x,(x-tmp)+ns);ns=tmp;values['neumaier'].append(ns+nc)
 record={'index':i,'set':label,'source_sha':item['sha'],'modes':{}}
 for name,seq in values.items():
  actual=torch.stack(seq,-1);delta=(actual-expected).double()
  m={'different_elements':int((actual.view(torch.uint8)!=expected.view(torch.uint8)).reshape(*actual.shape,-1).any(-1).sum()),'elements':actual.numel(),'max_abs':float(delta.abs().max()),'squared_error':float(delta.square().sum())}
  record['modes'][name]=m;ag=summary['modes'].setdefault(name,{'different_elements':0,'elements':0,'cases_equal':0,'max_abs':0.,'squared_error':0.})
  for field in ('different_elements','elements','squared_error'):ag[field]+=m[field]
  ag['cases_equal']+=int(m['different_elements']==0);ag['max_abs']=max(ag['max_abs'],m['max_abs'])
 summary['cases']+=1
 with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
summary['elapsed_s']=time.monotonic()-start
summary['script_sha']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
summary['inputs_freeze_sha']=hashlib.sha256((r/'torch-ut-g0-20261107/freeze.json').read_bytes()).hexdigest()
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
