"""One captured input: pinned Torch WY versus isolated blocked UT candidate."""
import ast, hashlib, json, time
from pathlib import Path
import torch, torch_npu
base=Path('/runtime'); out=base/'torch-ut-scripted-real-20261107';out.mkdir(exist_ok=True)
source=base/'runtime-wy-pinned.py'; ref=base/'ut-reference-20261107.py'
@torch.jit.script
def scripted_ut(attn: torch.Tensor, chunk_size: int) -> torch.Tensor:
 for row_idx in range(1, chunk_size):
  row = attn[..., row_idx, :row_idx].clone()
  sub = attn[..., :row_idx, :row_idx].clone()
  attn[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
 return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
ns={'torch':torch,'_ut_transform':scripted_ut}
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
torch.set_num_threads(1);torch.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
path=base/'wy921-localize-v1/capture/captures/rank0-call0001.pt'
saved=torch.load(path,map_location='cpu',weights_only=True);args=[x.npu() for x in saved['inputs']]
result={'baseline_sha':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_sha':hashlib.sha256(ref.read_bytes()).hexdigest(),'input_sha':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'single real-input diagnostic, not G0/G1 or performance acceptance'}
outputs={}
for label,fn in [('baseline',baseline),('scripted',ns['candidate'])]:
 for _ in range(5): fn(*args,64)
 torch.npu.synchronize();start=time.perf_counter()
 for _ in range(10): y=fn(*args,64)
 torch.npu.synchronize();result[label+'_ms']=(time.perf_counter()-start)*1000/10
 outputs[label]=[x.cpu() for x in y]
 (out/'result.json').write_text(json.dumps(result,indent=2))
result['comparisons']=[]
for name,a,b in zip(('q','k','w','u','g'),outputs['baseline'],outputs['scripted']):
 d=a.float()-b.float();result['comparisons'].append({'name':name,'equal':torch.equal(a,b),'different':int((a!=b).sum()),'max_abs':float(d.abs().max()),'relative_rmse':float(d.square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-30))})
result['speedup']=result['baseline_ms']/result['scripted_ms']
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
