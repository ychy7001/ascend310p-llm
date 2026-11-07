"""Paired operator diagnostic at 128/640 real and 3840 synthetic chunk shapes."""
import hashlib,json,statistics,time
from pathlib import Path
import torch,torch_npu
from torch_ut_graph_candidate import load_pair
base=Path('/runtime');out=base/'torch-ut-graph-long-20261107';source=base/'runtime-wy-pinned.py';baseline,candidate,graphs=load_pair(source)
assert hashlib.sha256(source.read_bytes()).hexdigest()=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
torch.set_num_threads(1)
summary={'scope':'operator timing diagnostic only, not model/production performance acceptance; 3840 token inputs repeat existing independent 64-token chunks and are synthetic','baseline_sha':hashlib.sha256(source.read_bytes()).hexdigest(),'candidate_sha':hashlib.sha256((base/'torch_ut_graph_candidate.py').read_bytes()).hexdigest(),'cases':[],'captures':graphs.captures}
for rank in (0,1):
 selected={}
 for path in sorted((base/'wy921-localize-v1/capture/captures').glob(f'rank{rank}-call*.pt')):
  values=torch.load(path,map_location='cpu',weights_only=True)['inputs'];tokens=values[0].shape[1]
  if tokens in (128,640) and tokens not in selected:selected[tokens]=(path,values)
  if len(selected)==2:break
 assert len(selected)==2
 torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 for tokens in (128,640,3840):
  path,values=selected[640 if tokens==3840 else tokens];repeat=6 if tokens==3840 else 1
  args=[x.repeat(*([1,repeat]+[1]*(x.ndim-2))).npu() if repeat>1 else x.npu() for x in values]
  expected=[x.cpu() for x in baseline(*args,64)];equal=[]
  for _ in range(3):equal.append([torch.equal(a.view(torch.uint8),b.cpu().view(torch.uint8)) for a,b in zip(expected,candidate(*args,64))])
  record={'rank':rank,'tokens':tokens,'source':path.name,'input_sha':hashlib.sha256(path.read_bytes()).hexdigest(),'repeat_factor':repeat,'synthetic':repeat>1,'byte_equal_repeats':equal}
  summary['cases'].append(record);(out/'result.json').write_text(json.dumps(summary,indent=2));assert all(all(v) for v in equal)
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
