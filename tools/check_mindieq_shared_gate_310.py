# SPDX-License-Identifier: Apache-2.0
"""Gate/Up token级量化共享：准备数值/异步/Graph和真实专家投影计时。"""
import argparse,json,sys,time
from pathlib import Path
from statistics import median
import torch
import torch_npu
from check_w4_repeatability_310 import load_bank
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experimental'))
from mindieq_activation import quantize_shared_routes
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True);p.add_argument('--library',required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(310);torch.ops.load_library(a.library)
bank=load_bank(a.model,('gate_proj','up_proj'));weight=bank[0].view(torch.int32).reshape(8,-1)
scale=bank[1].reshape(8,1024//16,2048//128,16)[:,:,0,:].reshape(8,1024).float().contiguous()
with a.out.open('x') as file:
 def emit(row):file.write(json.dumps(row)+'\n');file.flush();print(json.dumps(row),flush=True)
 emit({'kind':'config','scope':__doc__,'warmup':5,'repeats':30,'timing':'host dispatch+synchronize; paired alternating; preparation/full Gate projection','experts':8,'weights':'real layer0 expert0..7 gate+up','activations':'synthetic FP16','top_k':8})
 for tokens in (17,32,128,512):
  cpu=(torch.randn(tokens,2048)*.2).half();cpu[0]*=.0001
  x=cpu.npu();indices=torch.arange(tokens,dtype=torch.int64).repeat(8).npu()
  ends=torch.arange(1,9,dtype=torch.int64).mul(tokens).npu()
  def repeated():return torch.ops.qwen36_mindieq_grouped.quantize(x.index_select(0,indices).contiguous())
  def shared():return quantize_shared_routes(x,indices)
  def project(fn):
   q,s=fn();return torch.ops.qwen36_mindieq_grouped.matmul(q,s,weight,scale,ends)
  golden=[v.cpu() for v in repeated()]
  actual=[v.cpu() for v in shared()]
  assert all(torch.equal(a,b) for a,b in zip(actual,golden))
  reference=project(repeated).cpu();assert torch.equal(project(shared).cpu(),reference)
  pending=[project(shared) for _ in range(8)];torch.npu.synchronize()
  assert all(torch.equal(v.cpu(),reference) for v in pending)
  graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph):captured=project(shared)
  replacement=cpu.clone();replacement[0].fill_(.03125);x.copy_(replacement.npu())
  # 更新输入与动态路由索引，保持分组形状，检查Graph没有冻结量化输入。
  indices.copy_(torch.arange(tokens-1,-1,-1,dtype=torch.int64).repeat(8).npu())
  alternate=project(repeated).cpu();graph.replay();torch.npu.synchronize();assert torch.equal(captured.cpu(),alternate)
  x.copy_(cpu.npu());indices.copy_(torch.arange(tokens,dtype=torch.int64).repeat(8).npu());torch.npu.synchronize()
  samples={};functions={'prepare_repeated':repeated,'prepare_shared':shared,'gate_repeated':lambda:project(repeated),'gate_shared':lambda:project(shared)}
  for fn in functions.values():
   for _ in range(5):fn()
  torch.npu.synchronize()
  for name in functions:samples[name]=[]
  for repeat in range(30):
   for name in list(functions) if repeat%2==0 else list(reversed(functions)):
    begin=time.perf_counter();output=functions[name]();torch.npu.synchronize();samples[name].append((time.perf_counter()-begin)*1e3)
  emit({'kind':'case','tokens':tokens,'routed_rows':tokens*8,'status':'PASS','a8_codes_scale':'bitwise identical','gate_output':'bitwise identical','queued_outputs':8,'graph_changed_input_and_routes':'PASS','median_ms':{name:median(values) for name,values in samples.items()},'samples_ms':samples})
 emit({'kind':'summary','status':'PASS','cases':4})
