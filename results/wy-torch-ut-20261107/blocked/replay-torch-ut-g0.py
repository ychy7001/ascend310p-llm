"""One captured input: pinned Torch WY versus isolated blocked UT candidate."""
import ast, hashlib, json, time
from pathlib import Path
import torch, torch_npu
base=Path('/runtime'); out=base/'torch-ut-g0-20261107';out.mkdir(exist_ok=False)
source=base/'runtime-wy-pinned.py'; ref=base/'ut-reference-20261107.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
assert hashlib.sha256(ref.read_bytes()).hexdigest() == 'e024ecaee543964c44c6e80fa1c1ed5baaa1ec5e04df98c3555615bcbc19295d'
names={'_inv_small_unit_lower','_inv_unit_lower_recursive','_inv_unit_lower_triangular','_ut_transform'}
ns={'torch':torch,'_UT_INVERSE_BLOCK':8,'_UT_USE_BLOCKED_INVERSE':True}
refnodes=[x for x in ast.parse(ref.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
exec(compile(ast.Module(body=refnodes,type_ignores=[]),str(ref),'exec'),ns)
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
torch.set_num_threads(1)
items=[]
for label,folder in [('snapshots',base/'wy921-localize-v1/capture/captures'),('triggers',base/'wy4-candidate-v3/triggers')]:
 for path in sorted(folder.glob('rank*-call*.pt')):
  items.append((label,path,int(path.name[4]),None))
assert len(items)==132
selection=json.loads((base/'wy-guarded-20261106/boundary-selection.json').read_text());assert len(selection)==480
items += [('boundaries',base/'wy-boundary-20261106/capture'/r['file'],r['rank'],r['sha256']) for r in selection]
freeze={'baseline_sha':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_sha':hashlib.sha256(ref.read_bytes()).hexdigest(),'script_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'inputs':[{'set':label,'file':str(path.relative_to(base)),'sha':hashlib.sha256(path.read_bytes()).hexdigest()} for label,path,_,_ in items], 'scope':'612 fixed-input Torch blocked-UT G0; unchanged oracle/tolerances; no model/performance acceptance'}
(out/'freeze.json').write_text(json.dumps(freeze,indent=2))
for i,(label,path,rank,sha) in enumerate(items):
 if sha: assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 saved=torch.load(path,map_location='cpu',weights_only=False);args=[x.npu() for x in saved['inputs']]
 chunk=saved.get('chunk_size',64)
 original=[x.cpu() for x in baseline(*args,chunk)]
 actual=[x.cpu() for x in ns['candidate'](*args,chunk)]
 serial=[[torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(ns['candidate'](*args,chunk),actual)] for _ in range(3)]
 pending=[ns['candidate'](*args,chunk) for _ in range(3)]
 queued=[[torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(v,actual)] for v in pending]
 record={'index':i,'set':label,'file':str(path.relative_to(base)),'source_sha':freeze['inputs'][i]['sha'],'rank':rank,'serial':serial,'queued':queued,'repeat_pass':all(all(v) for v in serial+queued),'qkg_exact':all(torch.equal(original[j],actual[j]) for j in (0,1,4))}
 torch.save({'native_outputs':actual,'torch_outputs':original},out/f'{i:04d}.pt')
 with (out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
 if i%30==0:print(i+1,'/',len(items),flush=True)
 assert record['repeat_pass'] and record['qkg_exact'],record
(out/'device-summary.json').write_text(json.dumps({'cases':len(items),'repeat_passed':True,'qkg_exact':True,'status':'device_complete_requires_cpu'},indent=2))
