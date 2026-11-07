"""One captured input: pinned Torch WY versus isolated blocked UT candidate."""
import ast, hashlib, json, time
from pathlib import Path
import torch, torch_npu
base=Path('/runtime'); out=base/'torch-ut-no-clone-real-20261107';out.mkdir(exist_ok=True)
source=base/'runtime-wy-pinned.py'; ref=base/'ut-reference-20261107.py'
names={'_inv_small_unit_lower','_inv_unit_lower_recursive','_inv_unit_lower_triangular','_ut_transform'}
ns={'torch':torch,'_UT_INVERSE_BLOCK':8,'_UT_USE_BLOCKED_INVERSE':True}
refnodes=[x for x in ast.parse(ref.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
exec(compile(ast.Module(body=refnodes,type_ignores=[]),str(ref),'exec'),ns)
tree=ast.parse(source.read_text())
helpers=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in {'_expand_qk_to_v_heads','_compute_kernel_inputs_from_torch_wy'}]
exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),ns)
baseline=ns['_compute_kernel_inputs_from_torch_wy']
func=next(x for x in helpers if x.name=='_compute_kernel_inputs_from_torch_wy')
loop=next(x for x in func.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='row_idx')
changed=0
for node in loop.body[:2]:
 assert isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='clone'
 node.value=node.value.func.value
 changed+=1
assert changed==2
func.name='candidate'
exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(source),'exec'),ns)
torch.set_num_threads(1);torch.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
path=base/'wy921-localize-v1/capture/captures/rank0-call0001.pt'
saved=torch.load(path,map_location='cpu',weights_only=True);args=[x.npu() for x in saved['inputs']]
result={'baseline_sha':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_sha':hashlib.sha256(ref.read_bytes()).hexdigest(),'input_sha':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'single real-input diagnostic, not G0/G1 or performance acceptance'}
outputs={}
for label,fn in [('baseline',baseline),('no_clone',ns['candidate'])]:
 fn(*args,64);torch.npu.synchronize();start=time.perf_counter()
 for _ in range(3): y=fn(*args,64)
 torch.npu.synchronize();result[label+'_ms']=(time.perf_counter()-start)*1000/3
 outputs[label]=[x.cpu() for x in y]
 (out/'result.json').write_text(json.dumps(result,indent=2))
result['comparisons']=[]
for name,a,b in zip(('q','k','w','u','g'),outputs['baseline'],outputs['no_clone']):
 d=a.float()-b.float();result['comparisons'].append({'name':name,'equal':torch.equal(a,b),'different':int((a!=b).sum()),'max_abs':float(d.abs().max()),'relative_rmse':float(d.square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-30))})
result['speedup']=result['baseline_ms']/result['no_clone_ms']
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
