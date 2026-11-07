"""Single bounded whole-WY graph diagnostic; preserve all original Torch ops."""
import ast,hashlib,json,statistics,time
from pathlib import Path
import torch,torch_npu
base=Path('/runtime');out=base/'torch-wy-fullgraph-20261107';source=base/'runtime-wy-pinned.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
ns={'torch':torch};helpers=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'_expand_qk_to_v_heads','_compute_kernel_inputs_from_torch_wy'}];exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),ns);baseline=ns['_compute_kernel_inputs_from_torch_wy']
class GraphWY:
 def __init__(self):self.cache={};self.captures=[]
 def __call__(self,*args):
  tensors=args[:-1];chunk=args[-1];key=(tuple((tuple(x.shape),tuple(x.stride()),str(x.dtype),str(x.device)) for x in tensors),chunk)
  if key not in self.cache:
   assert len(self.cache)<4 and not torch.npu.is_current_stream_capturing()
   static=[torch.empty_like(x) for x in tensors];stream=torch.npu.Stream();current=torch.npu.current_stream();stream.wait_stream(current)
   with torch.npu.stream(stream):
    for _ in range(3):
     for a,b in zip(static,tensors):a.copy_(b)
     baseline(*static,chunk)
   current.wait_stream(stream);torch.npu.synchronize()
   for a,b in zip(static,tensors):a.copy_(b)
   torch.npu.synchronize();graph=torch.npu.NPUGraph();start=time.perf_counter()
   with torch.npu.graph(graph,stream=stream):outputs=baseline(*static,chunk)
   current.wait_stream(stream);torch.npu.synchronize();self.cache[key]=(graph,static,outputs);self.captures.append({'device':str(tensors[0].device),'tokens':tensors[0].shape[1],'capture_s':time.perf_counter()-start,'allocated_bytes':torch.npu.memory_allocated(),'reserved_bytes':torch.npu.memory_reserved()});assert torch.npu.memory_allocated()<2*1024**3
  graph,static,outputs=self.cache[key]
  for a,b in zip(static,tensors):a.copy_(b)
  graph.replay()
  return tuple(x.clone() for x in outputs)
candidate=GraphWY();torch.set_num_threads(1)
summary={'scope':'complete original WY graph operator diagnostic; preserves arithmetic; includes five input copies and five output clones; 3840 synthetic from repeated640 chunks; no model/performance adoption gate','baseline_sha':hashlib.sha256(source.read_bytes()).hexdigest(),'probe_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[],'captures':candidate.captures}
for rank in (0,1):
 selected=[]
 for path in sorted((base/'wy921-localize-v1/capture/captures').glob(f'rank{rank}-call*.pt')):
  inputs=torch.load(path,map_location='cpu',weights_only=True)['inputs']
  if inputs[0].shape[1]==640:selected.append((path,inputs))
  if len(selected)==2:break
 assert len(selected)==2;torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 for tokens in (640,3840):
  repeat=6 if tokens==3840 else 1;all_args=[];expected=[]
  for _,values in selected:
   args=[x.repeat(*([1,repeat]+[1]*(x.ndim-2))).npu() if repeat>1 else x.npu() for x in values];all_args.append(args);expected.append([x.cpu() for x in baseline(*args,64)])
  serial=[]
  for index in (0,1,0):serial.append([torch.equal(a.view(torch.uint8),b.cpu().view(torch.uint8)) for a,b in zip(expected[index],candidate(*all_args[index],64))])
  pending=[candidate(*all_args[index],64) for index in (0,1,0)];queued=[[torch.equal(a.view(torch.uint8),b.cpu().view(torch.uint8)) for a,b in zip(expected[index],values)] for index,values in zip((0,1,0),pending)]
  record={'rank':rank,'tokens':tokens,'synthetic':repeat>1,'repeat_factor':repeat,'inputs':[{'file':p.name,'sha':hashlib.sha256(p.read_bytes()).hexdigest()} for p,_ in selected],'serial_byte_equal':serial,'queued_alternating_inputs_byte_equal':queued};summary['cases'].append(record);(out/'result.json').write_text(json.dumps(summary,indent=2));assert all(all(v) for v in serial+queued)
  args=all_args[0]
  for _ in range(5):baseline(*args,64);candidate(*args,64)
  pairs=[]
  for turn in range(8):
   times={}
   for label,fn in ([('baseline',baseline),('graph',candidate)] if turn%2==0 else [('graph',candidate),('baseline',baseline)]):
    torch.npu.synchronize();start=time.perf_counter()
    for _ in range(10):fn(*args,64)
    torch.npu.synchronize();times[label]=(time.perf_counter()-start)*1000/10
   pairs.append({**times,'speedup':times['baseline']/times['graph']})
  record['timing_pairs']=pairs;record['median_speedup']=statistics.median(p['speedup'] for p in pairs);record['faster_pairs']=sum(p['speedup']>1 for p in pairs);(out/'result.json').write_text(json.dumps(summary,indent=2))
summary['completed']=True;(out/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
