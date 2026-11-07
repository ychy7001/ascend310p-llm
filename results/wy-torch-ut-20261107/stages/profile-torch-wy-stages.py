"""Synchronised stage diagnostic from pinned WY source; not an acceptance benchmark."""
import ast,hashlib,json,time,statistics,copy
from pathlib import Path
import torch,torch_npu
base=Path('/runtime');out=base/'torch-wy-stages-20261107';out.mkdir(exist_ok=False)
p=base/'runtime-wy-pinned.py';raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
assert sha=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
nodes=[x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name in {'_expand_qk_to_v_heads','_compute_kernel_inputs_from_torch_wy'}]
ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
baseline=ns['_compute_kernel_inputs_from_torch_wy']; fn=copy.deepcopy(nodes[-1]);assert fn.name=='_compute_kernel_inputs_from_torch_wy'
# Mark stage boundaries without changing any mathematical statement.
body=[];markers=[]
for node in fn.body:
 label=None
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  target=node.targets[0].id
  if target=='q_kernel':label='layout_and_cumsum'
  elif target=='lower_decay':label='decay'
  elif target=='k_beta':label='gram'
  elif target=='mask_diag':label='mask'
  elif target=='value' and isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.MatMult):label='wu_apply'
  elif target=='u_kernel':label='output_cast'
 if isinstance(node,ast.For):label='ut_rows_and_identity'
 if isinstance(node,ast.Return):label='end'
 if label:
  body.extend(ast.parse(f'mark({label!r})').body);markers.append(label)
 body.append(node)
assert markers==['layout_and_cumsum','decay','gram','mask','ut_rows_and_identity','wu_apply','output_cast','end'],markers
fn.body=body;fn.name='profiled';trace=[];state=[None,None]
def mark(label):
 torch.npu.synchronize();now=time.perf_counter()
 if state[0] is not None:trace.append((state[0],(now-state[1])*1000))
 state[:]=[label,now]
ns['mark']=mark;exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns)
torch.set_num_threads(1);torch.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
captures=base/'wy921-localize-v1/capture/captures'
# Use two deterministic real captures. No new model requests or sampling.
paths=[captures/'rank0-call0001.pt',captures/'rank0-call0060.pt'];rows=[]
for path in paths:
 saved=torch.load(path,map_location='cpu',weights_only=True);args=[x.npu() for x in saved['inputs']]
 expected=[x.cpu() for x in baseline(*args,64)]
 plain=[];stages=[]
 for rep in range(3):
  torch.npu.synchronize();started=time.perf_counter();actual=baseline(*args,64);torch.npu.synchronize();plain.append((time.perf_counter()-started)*1000)
  trace.clear();state[:]=[None,None];profiled=ns['profiled'](*args,64)
  equal=[torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(profiled,expected)]
  assert all(equal),equal
  stages.append(dict(trace))
 row={'file':path.name,'input_sha':hashlib.sha256(path.read_bytes()).hexdigest(),'shapes':[list(x.shape) for x in args],'plain_ms':plain,'stage_median_ms':{k:statistics.median(x[k] for x in stages) for k in stages[0]},'stage_samples_ms':stages,'all_outputs_byte_equal':True}
 rows.append(row);print(json.dumps(row),flush=True)
result={'source_sha':sha,'cases':rows,'scope':'synchronised WY stages; boundaries perturb scheduling, stage times are not additive estimates of production time; no G0/G1/performance acceptance'}
(out/'result.json').write_text(json.dumps(result,indent=2))
