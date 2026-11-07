"""Experimental exact-row UT graph scheduler; no native math changes."""
import ast,time
import torch

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
   assert sum(k[2]==key[2] for k in self.cache)<16,'fixed-set per-device graph limit'
   assert not torch.npu.is_current_stream_capturing(),'nested graph capture forbidden'
   static=torch.empty_like(attn);stream=torch.npu.Stream();current=torch.npu.current_stream();stream.wait_stream(current)
   with torch.npu.stream(stream):
    for _ in range(3):static.copy_(attn);rows(static,chunk_size)
   current.wait_stream(stream);torch.npu.synchronize();static.copy_(attn);torch.npu.synchronize()
   graph=torch.npu.NPUGraph();start=time.perf_counter()
   with torch.npu.graph(graph,stream=stream):result=rows(static,chunk_size)
   current.wait_stream(stream);torch.npu.synchronize()
   self.cache[key]=(graph,static,result)
   assert torch.npu.memory_allocated()<=2*1024**3,'fixed-set device allocation budget'
   self.captures.append({'device':str(attn.device),'shape':list(attn.shape),'capture_s':time.perf_counter()-start,'allocated_bytes':torch.npu.memory_allocated(),'reserved_bytes':torch.npu.memory_reserved()})
  graph,static,result=self.cache[key];static.copy_(attn);graph.replay();return result

def load_pair(source):
 graph_ut=GraphUT();ns={'torch':torch,'_ut_transform':graph_ut};tree=ast.parse(source.read_text());helpers=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in {'_expand_qk_to_v_heads','_compute_kernel_inputs_from_torch_wy'}]
 exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),ns);baseline=ns['_compute_kernel_inputs_from_torch_wy'];func=next(x for x in helpers if x.name=='_compute_kernel_inputs_from_torch_wy');idx=next(i for i,x in enumerate(func.body) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='row_idx');assert isinstance(func.body[idx+1],ast.Assign)
 func.body[idx:idx+2]=ast.parse('attn = _ut_transform(attn, chunk_size)').body;func.name='candidate';exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(source),'exec'),ns);candidate=ns['candidate']
 return baseline,candidate,graph_ut
