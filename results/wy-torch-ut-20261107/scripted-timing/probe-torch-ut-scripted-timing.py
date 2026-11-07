"""Bounded alternating-order timing of exact-row scripted UT on existing captures."""
import ast, hashlib, json, time
from pathlib import Path
import torch, torch_npu
base=Path('/runtime'); out=base/'torch-ut-scripted-timing-20261107';out.mkdir(exist_ok=False)
source=base/'runtime-wy-pinned.py'; ref=base/'ut-reference-20261107.py'
@torch.jit.script
def scripted_ut(attn: torch.Tensor, chunk_size: int) -> torch.Tensor:
 for row_idx in range(1, chunk_size):
  row = attn[..., row_idx, :row_idx].clone()
  sub = attn[..., :row_idx, :row_idx].clone()
  attn[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
 return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
ns={'torch':torch,'_ut_transform':scripted_ut}
assert hashlib.sha256(source.read_bytes()).hexdigest() == '4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
tree=ast.parse(source.read_text())
helpers=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in {'_expand_qk_to_v_heads','_compute_kernel_inputs_from_torch_wy'}]
exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),ns)
baseline=ns['_compute_kernel_inputs_from_torch_wy']
func=next(x for x in helpers if x.name=='_compute_kernel_inputs_from_torch_wy')
idx=next(i for i,x in enumerate(func.body) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='row_idx')
assert isinstance(func.body[idx+1],ast.Assign)
func.body[idx:idx+2]=ast.parse('attn = _ut_transform(attn, chunk_size)').body
func.name='candidate'
exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(source),'exec'),ns)

import statistics

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
torch.set_num_threads(1)
# Fixed deterministic selection: up to three distinct shapes per rank from the
# existing 120 snapshots. No synthetic inputs or newly captured model requests.
representatives={}
for path in sorted((base/'wy921-localize-v1/capture/captures').glob('rank*-call*.pt')):
 saved=torch.load(path,map_location='cpu',weights_only=False)
 rank=int(path.name[4]);shape=tuple(tuple(x.shape) for x in saved['inputs'])
 representatives.setdefault((rank,shape),path)
selected=[]
for rank in (0,1):
 keys=sorted([k for k in representatives if k[0]==rank],key=lambda k:k[1])
 indices=sorted({0,len(keys)//2,len(keys)-1})
 selected += [(rank,representatives[keys[i]],keys[i][1]) for i in indices]
freeze={'script_sha':digest(Path(__file__)),'baseline_sha':digest(source),
 'scope':'alternating microdiagnostic, not model or Prefill acceptance',
 'pairs':8,'calls_per_arm':10,'warmups_per_arm':5,
 'selection':[{'rank':rank,'file':str(path.relative_to(base)),'sha':digest(path),'shapes':shape} for rank,path,shape in selected]}
(out/'freeze.json').write_text(json.dumps(freeze,indent=2))
rows=[]
for rank,path,shape in selected:
 torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 saved=torch.load(path,map_location='cpu',weights_only=False)
 args=[x.npu() for x in saved['inputs']];chunk=saved.get('chunk_size',64)
 functions={'baseline':baseline,'scripted':ns['candidate']}
 for fn in functions.values():
  for _ in range(5):fn(*args,chunk)
 torch.npu.synchronize()
 pairs=[]
 for pair in range(8):
  row={'pair':pair,'order':['baseline','scripted'] if pair%2==0 else ['scripted','baseline']}
  values={}
  for label in row['order']:
   fn=functions[label];torch.npu.synchronize();start=time.perf_counter()
   for _ in range(10):actual=fn(*args,chunk)
   torch.npu.synchronize();row[label+'_ms']=(time.perf_counter()-start)*100
   values[label]=[x.cpu() for x in actual]
  row['byte_equal']=all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(values['baseline'],values['scripted']))
  assert row['byte_equal'],row
  row['speedup']=row['baseline_ms']/row['scripted_ms'];pairs.append(row)
 result={'rank':rank,'file':str(path.relative_to(base)),'shapes':shape,'pairs':pairs,
  'median_speedup':statistics.median(p['speedup'] for p in pairs),
  'baseline_median_ms':statistics.median(p['baseline_ms'] for p in pairs),
  'scripted_median_ms':statistics.median(p['scripted_ms'] for p in pairs),
  'scripted_faster_pairs':sum(p['speedup']>1 for p in pairs)}
 rows.append(result);(out/'result.json').write_text(json.dumps({'scope':freeze['scope'],'cases':rows},indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','shapes')}),flush=True)
