# SPDX-License-Identifier: Apache-2.0
"""长C8路径拆分：单层Q1转换、native PA、组合；不能外推MTP整模型速度。"""
import argparse,json,time
from pathlib import Path
from statistics import median
import torch
import torch_npu
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(310)
torch.ops.load_library(a.library);torch.npu.set_compile_mode(jit_compile=False)
with a.out.open('x') as f:
 def emit(row):f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
 emit({'kind':'config','scope':__doc__,'warmup':5,'repeats':20,'timing':'paired alternating host Graph replay+synchronize, preallocated outputs','heads':16,'kv_heads':2,'dim':256,'query_tokens':1})
 for length in (8192,32768,114688):
  kcpu=torch.randint(-127,128,(length,2,256),dtype=torch.int8)
  vcpu=torch.randint_like(kcpu,-127,128)
  scale=torch.ones(length,16,dtype=torch.float16)*.01
  k=kcpu.npu();v=vcpu.npu();ks=scale.npu();vs=scale.npu()
  keys=(kcpu.half()*scale[:,:2,None]).npu();values=(vcpu.half()*scale[:,:2,None]).npu()
  shape=(length//64,32,64,16)
  nk=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
  nv=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
  ck=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
  cv=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
  table=torch.arange(length//64,dtype=torch.int32).npu();slots=torch.arange(length,dtype=torch.int32).npu()
  context=torch.tensor([length],dtype=torch.int32).npu()
  query=torch.randn(1,16,256).half().npu();no=torch.empty_like(query);co=torch.empty_like(query)
  torch_npu._npu_reshape_and_cache(key=keys,value=values,key_cache=nk,value_cache=nv,slot_indices=slots)
  def convert():torch.ops.npu.c8_cache_to_nz(k,v,ks,vs,ck,cv,64,table,context)
  def pa(key,value,out):
   torch_npu._npu_paged_attention(query=query,key_cache=key,value_cache=value,num_kv_heads=2,num_heads=16,scale_value=256**-.5,block_table=table.reshape(1,-1),context_lens=context,out=out)
  def native():pa(nk,nv,no)
  def staged():convert();pa(ck,cv,co)
  native();staged();torch.npu.synchronize()
  torch.testing.assert_close(ck.cpu(),nk.cpu(),rtol=0,atol=0)
  torch.testing.assert_close(cv.cpu(),nv.cpu(),rtol=0,atol=0)
  torch.testing.assert_close(co.cpu(),no.cpu(),rtol=0,atol=0)
  funcs={'conversion':convert,'native_pa':native,'staged':staged};graphs={}
  for name,fn in funcs.items():
   for _ in range(5):fn()
   torch.npu.synchronize();graph=torch.npu.NPUGraph()
   with torch.npu.graph(graph):fn()
   graphs[name]=graph
  for graph in graphs.values():
   for _ in range(5):graph.replay()
  torch.npu.synchronize();samples={name:[] for name in graphs}
  for repeat in range(20):
   for name in list(graphs) if repeat%2==0 else list(reversed(graphs)):
    begin=time.perf_counter();graphs[name].replay();torch.npu.synchronize();samples[name].append((time.perf_counter()-begin)*1e3)
  emit({'kind':'measurement','tokens':length,'status':'PASS','cache_and_attention_vs_fp16':'bitwise identical after C8 reconstruction','median_ms':{name:median(values) for name,values in samples.items()},'samples_ms':samples,'c8_storage_bytes':sum(t.numel()*t.element_size() for t in (k,v,ks,vs)),'scratch_bytes':(ck.numel()+cv.numel())*2,'scope':'synthetic C8 data, identity block table, Q1 native PA; no model/router/MTP'})
 emit({'kind':'summary','status':'PASS','cases':3})
