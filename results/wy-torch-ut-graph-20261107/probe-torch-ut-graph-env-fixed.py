"""Isolated NPU-graph UT feasibility, exact original row arithmetic, four inputs."""
import ast,hashlib,json,statistics,time
from pathlib import Path
import torch,torch_npu
base=Path('/runtime');out=base/'torch-ut-graph-env-fixed-20261107';source=base/'runtime-wy-pinned.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
torch.set_num_threads(1);torch.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)

def rows(attn,chunk_size):
 for row_idx in range(1,chunk_size):
  row=attn[...,row_idx,:row_idx].clone()
  sub=attn[...,:row_idx,:row_idx].clone()
  attn[...,row_idx,:row_idx]=row+(row.unsqueeze(-1)*sub).sum(-2)
 return attn+torch.eye(chunk_size,dtype=attn.dtype,device=attn.device)

class GraphUT:
 def __init__(self):self.cache={};self.captures=[]
 def __call__(self,attn,chunk_size):
  key=(tuple(attn.shape),str(attn.dtype),str(attn.device),tuple(attn.stride()))
  if key not in self.cache:
   assert len(self.cache)<2,'bounded pilot expects two shapes only'
   static=torch.empty_like(attn);stream=torch.npu.Stream();current=torch.npu.current_stream();stream.wait_stream(current)
   with torch.npu.stream(stream):
    for _ in range(3):static.copy_(attn);rows(static,chunk_size)
   current.wait_stream(stream);torch.npu.synchronize();static.copy_(attn);torch.npu.synchronize()
   graph=torch.npu.NPUGraph();start=time.perf_counter()
   with torch.npu.graph(graph,stream=stream):result=rows(static,chunk_size)
   current.wait_stream(stream);torch.npu.synchronize()
   self.cache[key]=(graph,static,result)
   self.captures.append({'shape':list(attn.shape),'capture_s':time.perf_counter()-start,'allocated_bytes':torch.npu.memory_allocated(),'reserved_bytes':torch.npu.memory_reserved()})
  graph,static,result=self.cache[key];static.copy_(attn);graph.replay();return result

graph_ut=GraphUT();ns={'torch':torch,'_ut_transform':graph_ut};tree=ast.parse(source.read_text());helpers=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in {'_expand_qk_to_v_heads','_compute_kernel_inputs_from_torch_wy'}]
exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),ns);baseline=ns['_compute_kernel_inputs_from_torch_wy'];func=next(x for x in helpers if x.name=='_compute_kernel_inputs_from_torch_wy');idx=next(i for i,x in enumerate(func.body) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='row_idx');assert isinstance(func.body[idx+1],ast.Assign)
func.body[idx:idx+2]=ast.parse('attn = _ut_transform(attn, chunk_size)').body;func.name='candidate';exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(source),'exec'),ns);candidate=ns['candidate']
items=[]
for rank in (0,1):
 selected={}
 for path in sorted((base/'wy921-localize-v1/capture/captures').glob(f'rank{rank}-call*.pt')):
  saved=torch.load(path,map_location='cpu',weights_only=True);tokens=saved['inputs'][0].shape[1]
  if tokens in (128,640) and tokens not in selected:selected[tokens]=(path,saved['inputs'])
  if len(selected)==2:break
 assert len(selected)==2
 items.extend(selected[t] for t in (128,640))
result={'scope':'four fixed real inputs, graph API feasibility and byte identity only; paired timing is diagnostic, not performance/adoption acceptance','baseline_sha':hashlib.sha256(source.read_bytes()).hexdigest(),'probe_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[],'captures':graph_ut.captures}
for path,inputs in items:
 args=[x.npu() for x in inputs];expected=[x.cpu() for x in baseline(*args,64)];comparisons=[]
 for _ in range(3):comparisons.append([torch.equal(a,b.cpu()) for a,b in zip(expected,candidate(*args,64))])
 result['cases'].append({'file':path.name,'input_sha':hashlib.sha256(path.read_bytes()).hexdigest(),'tokens':inputs[0].shape[1],'byte_equal_repeats':comparisons})
 (out/'result.json').write_text(json.dumps(result,indent=2));assert all(all(v) for v in comparisons),'graph changes output'
for record,(path,inputs) in zip(result['cases'],items):
 args=[x.npu() for x in inputs]
 for _ in range(5):baseline(*args,64);candidate(*args,64)
 pairs=[]
 for turn in range(8):
  times={}
  for label,fn in ([('baseline',baseline),('graph',candidate)] if turn%2==0 else [('graph',candidate),('baseline',baseline)]):
   torch.npu.synchronize();start=time.perf_counter()
   for _ in range(10):fn(*args,64)
   torch.npu.synchronize();times[label]=(time.perf_counter()-start)*1000/10
  pairs.append({**times,'speedup':times['baseline']/times['graph']})
 record['timing_pairs']=pairs;record['median_speedup']=statistics.median(v['speedup'] for v in pairs);record['faster_pairs']=sum(v['speedup']>1 for v in pairs)
 (out/'result.json').write_text(json.dumps(result,indent=2))
result['pilot_passed']=all(all(all(v) for v in c['byte_equal_repeats']) for c in result['cases']);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
