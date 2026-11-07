"""CPU numerical diagnosis on frozen captures; does not change any gate."""
import hashlib,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'/kernel')
from wy_snapshot_reference import reference
r=Path('/work');out=r/'wy-fp64-gram-rounding-20261107';out.mkdir(exist_ok=False)
torch.set_num_threads(4);started=time.monotonic()
freeze=json.loads((r/'torch-ut-g0-20261107/freeze.json').read_text())
assert len(freeze['inputs'])==612

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def coefficients(inputs,dtype):
 q,k,v,g,beta=inputs;b,t,hk,kd=k.shape;hv,vd=v.shape[2:]
 key=k.to(dtype).repeat_interleave(hv//hk,2).transpose(1,2).reshape(b,hv,-1,64,kd)
 value=v.to(dtype).transpose(1,2).reshape(b,hv,-1,64,vd)
 gate=g.to(dtype).transpose(1,2).reshape(b,hv,-1,64).cumsum(-1)
 weight=beta.to(dtype).transpose(1,2).reshape(b,hv,-1,64)
 decay=torch.exp(torch.tril(gate.unsqueeze(-1)-gate.unsqueeze(-2))).tril()
 matrix=torch.tril((key*weight.unsqueeze(-1))@key.transpose(-1,-2)*decay,diagonal=-1)+torch.eye(64,dtype=dtype)
 return matrix,key*weight.unsqueeze(-1)*gate.exp().unsqueeze(-1),value*weight.unsqueeze(-1),(b,hv,t,kd,vd)
def solve(c,half_u=False):
 matrix,w,u,shape=c;b,h,t,kd,vd=shape
 if half_u:u=u.half().double()
 return [torch.linalg.solve_triangular(matrix.double(),x.double(),upper=False,unitriangular=True).reshape(b,h,t,d) for x,d in ((w,kd),(u,vd))]
def gram_model(inputs,gate_device,fold_beta):
 q,k,v,g,beta=inputs;b,t,hk,kd=k.shape;hv,vd=v.shape[2:]
 key=k.float().repeat_interleave(hv//hk,2).transpose(1,2).reshape(b,hv,-1,64,kd)
 value=v.float().transpose(1,2).reshape(b,hv,-1,64,vd)
 weight=beta.float().transpose(1,2).reshape(b,hv,-1,64)
 gate=gate_device.float().reshape(b,hv,-1,64)
 kbeta=key*weight.unsqueeze(-1)
 lhs=kbeta.half().float() if fold_beta else key
 gram=sum(lhs[...,n:n+64]@key[...,n:n+64].transpose(-1,-2) for n in range(0,kd,64))
 if not fold_beta:gram=gram*weight.unsqueeze(-1)
 decay=torch.exp(torch.tril(gate.unsqueeze(-1)-gate.unsqueeze(-2))).tril()
 matrix=torch.tril(gram*decay,diagonal=-1)+torch.eye(64)
 return matrix,kbeta*gate.exp().unsqueeze(-1),value*weight.unsqueeze(-1),(b,hv,t,kd,vd)

def measure(a,e):
 a=a.double();e=e.double();delta=a-e;error=delta.abs();tol=0.0005+0.04*e.abs()
 where=int(error.argmax());idx=[];offset=where
 for size in reversed(a.shape):idx.append(offset%size);offset//=size
 return {'squared_error':float(delta.square().sum()),'elements':a.numel(),'outside':int((error>tol).sum()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(e).all()),'max_abs':float(error.flatten()[where]),'coordinate':list(reversed(idx)),'actual_at_max':float(a.flatten()[where]),'expected_at_max':float(e.flatten()[where]),'relative_rmse':float(delta.square().mean().sqrt()/e.square().mean().sqrt().clamp_min(1e-30))}
summary={'scope':'CPU Gram-rounding model only; actual gate from previously rejected library-Gram output; double solve and final half; not a device implementation or gate','sets':{},'elapsed_s':None}
provenance={'script_sha':digest(Path(__file__)),'fixed_inputs_sha':digest(r/'torch-ut-g0-20261107/freeze.json'),'library_gram_freeze_sha':digest(r/'gram-g0/freeze.json'),'inputs':[]}
for i,item in enumerate(freeze['inputs']):
 label=item['set'];rel=Path(item['file'])
 if label=='snapshots':path=r/'cpu-old/wy921-localize-20261104/raw/captures'/rel.name;device=r/'gram-g0/wy/snapshots'/rel.name
 elif label=='triggers':path=r/'cpu-old/wy921-fix-20261105/raw/triggers'/rel.name;device=r/'gram-g0/wy/triggers'/rel.name
 else:path=r/'cpu-boundary'/Path(*rel.parts[2:]);device=r/'gram-g0/boundaries'/f'{i-132:04d}.pt'
 assert digest(path)==item['sha'],path
 saved=torch.load(path,map_location='cpu',weights_only=False);inputs=saved['inputs']
 data=torch.load(device,map_location='cpu',weights_only=True)
 c64=coefficients(inputs,torch.float64);c32=coefficients(inputs,torch.float32)
 ideal=solve(c64);matched=solve(c32)
 fp32=reference(inputs)[2:4]
 halfgram=solve(gram_model(inputs,data['native_outputs'][4],True))
 postbeta=solve(gram_model(inputs,data['native_outputs'][4],False))
 modes={'device_torch':data['torch_outputs'][2:4],
 'device_library_gram':data['native_outputs'][2:4],
 'model_half_kbeta_gram_final_half':[x.half().double() for x in halfgram],
 'model_raw_k_gram_then_beta_final_half':[x.half().double() for x in postbeta]}
 group=summary['sets'].setdefault(label,{'cases':0,'comparisons':{}});group['cases']+=1
 row={'index':i,'set':label,'file':item['file'],'source_sha':item['sha'],'device_output_sha':digest(device),'comparisons':{}}
 provenance['inputs'].append({k:row[k] for k in ('index','source_sha','device_output_sha')})
 for refname,expected in [('full_fp64',ideal),('device_library_gram',data['native_outputs'][2:4])]:
  for name,values in modes.items():
   key=refname+'/'+name;row['comparisons'][key]={}
   ag=group['comparisons'].setdefault(key,{'w':{'squared_error':0.,'outside':0,'elements':0,'cases_outside':0,'max_abs':0.},'u':{'squared_error':0.,'outside':0,'elements':0,'cases_outside':0,'max_abs':0.}})
   for output,a,e in zip(('w','u'),values,expected):
    m=measure(a,e);assert m['finite'],(i,key,output)
    row['comparisons'][key][output]=m
    for field in ('squared_error','outside','elements'):ag[output][field]+=m[field]
    ag[output]['cases_outside']+=int(m['outside']>0)
    ag[output]['max_abs']=max(ag[output]['max_abs'],m['max_abs'])
 with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if (i+1)%60==0:print(i+1,'/612',flush=True)
summary['elapsed_s']=time.monotonic()-started
(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'sets':{k:v['cases'] for k,v in summary['sets'].items()},'elapsed_s':summary['elapsed_s']}),flush=True)
