"""Localize the remaining four U outliers without a device window."""
import ast,hashlib,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'/kernel')
from wy_snapshot_reference import reference
r=Path('/work');out=r/'postbeta-residual-20261107';out.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
helper=r/'analyze-wy-fp64-gram-rounding.py'
nodes=[n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'coefficients','solve','gram_model'}]
ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper),'exec'),ns)
freeze=json.loads((r/'torch-ut-g0-20261107/freeze.json').read_text())
metrics=[json.loads(s) for s in (r/'postbeta-cpu/rows.jsonl').read_text().splitlines()]
rows=[]
for i,record in enumerate(metrics):
 if all(m['passed'] for m in record['native_vs_cpu']):continue
 item=freeze['inputs'][i];label=item['set'];rel=Path(item['file'])
 if label=='snapshots':path=r/'cpu-old/wy921-localize-20261104/raw/captures'/rel.name;device=r/'postbeta-g0/wy/snapshots'/rel.name
 elif label=='triggers':path=r/'cpu-old/wy921-fix-20261105/raw/triggers'/rel.name;device=r/'postbeta-g0/wy/triggers'/rel.name
 else:path=r/'cpu-boundary'/Path(*rel.parts[2:]);device=r/'postbeta-g0/boundaries'/f'{i-132:04d}.pt'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha']
 saved=torch.load(path,map_location='cpu',weights_only=False);data=torch.load(device,map_location='cpu',weights_only=True)
 inputs=saved['inputs'];expected=reference(inputs)[3];actual=data['native_outputs'][3]
 matrix,w,u,shape=ns['gram_model'](inputs,data['native_outputs'][4],False)
 full=ns['solve']((matrix,w,u,shape))[1].half().double()
 half=ns['solve']((matrix,w,u,shape),half_u=True)[1].half().double()
 outside=((actual.float()-expected).abs()>0.0005+0.04*expected.abs()).nonzero()
 for coord in outside:
  b,h,t,d=map(int,coord);chunk=t//64;row=t%64;a=matrix[b,h,chunk]-torch.eye(64)
  max_row_sum=float(a.abs().sum(-1).max());frob=float(a.square().sum())
  fast=(max_row_sum<0.75 or frob<7.502183356513852)
  key=(b,h,t,d);tol=float(0.0005+0.04*expected[key].abs())
  rows.append({'index':i,'set':label,'source_sha':item['sha'],'file':item['file'],'coordinate':list(key),'chunk':chunk,'chunk_row':row,
   'native':float(actual[key]),'cpu_fp32':float(expected[key]),'tolerance':tol,
   'native_error':float((actual[key].float()-expected[key]).abs()),
   'postbeta_full_rhs_double_solve_final_half':float(full[key]),'postbeta_half_rhs_double_solve_final_half':float(half[key]),
   'full_rhs_error':float((full[key]-expected[key]).abs()),'half_rhs_error':float((half[key]-expected[key]).abs()),
   'max_row_sum_cpu_model':max_row_sum,'frob_squared_cpu_model':frob,'predicted_route':'fast_doubling' if fast else 'fp32_refined',
   'route_note':'derived from CPU Gram model; not instrumented device branch trace'})
assert len(rows)==4,rows
summary={'scope':'4 existing outliers, CPU route/RHS model only','rows':rows,'elapsed_s':time.monotonic()-start,'script_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
