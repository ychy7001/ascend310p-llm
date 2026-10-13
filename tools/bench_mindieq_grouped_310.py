# SPDX-License-Identifier: Apache-2.0
"""单launch MindIEQ grouped原型：整数精度、动态专家边界、Graph及三真实counts。"""
import argparse,importlib.util,json,time
from pathlib import Path
from itertools import accumulate
from statistics import median
import torch
import torch_npu
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True);p.add_argument('--upstream-test',required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--route-trace',type=Path)
p.add_argument('--repeats',type=int,default=20)
p.add_argument('--cases',nargs='+',help='仅执行指定案例，默认全部')
a=p.parse_args();torch.manual_seed(20261013);torch.set_num_threads(4)
torch.ops.load_library(a.library)
spec=importlib.util.spec_from_file_location('qmm',a.upstream_test);ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
cases=[('tail_zero',[1,0,2,15,16,17,63,129]),('balanced',[128]*8),('one_expert',[4096]+[0]*7)]
if a.route_trace:
 traces=[json.loads(s) for s in a.route_trace.read_text().splitlines()];traces=[r for r in traces if r['chunk_tokens']==512];traces.sort(key=lambda r:max(r['counts']))
 cases += [('real_counts_'+str(percent),traces[int((len(traces)-1)*percent)]['counts']) for percent in [.05,.5,.95]]
if a.cases:cases=[case for case in cases if case[0] in a.cases]
if not cases:raise ValueError('no matching case')
with a.out.open('x') as f:
 f.write(json.dumps({'kind':'config','scope':__doc__,'warmup':5,'repeats':a.repeats,'weights':'synthetic integer W4; activations synthetic integer A8; actual shapes','padding_rows':16})+'\n');f.flush()
 for k,n in [(2048,1024),(512,2048)]:
  for label,counts in cases:
   e=len(counts);m=sum(counts)
   wc=torch.randint(-8,8,(e,n,k),dtype=torch.int8);ws_cpu=torch.rand(e,n)*.01
   packed=torch.stack([torch.from_numpy(ext.pack_biased_nz_int32_vec(w.T.contiguous().numpy(),k,n).copy()) for w in wc]).npu();ws=ws_cpu.npu()
   xcpu=torch.randint(-127,128,(m+16,k),dtype=torch.int8);scpu=torch.rand(m+16)*.01
   x=xcpu.npu();scale=scpu.npu();ends_cpu=list(accumulate(counts));ends=torch.tensor(ends_cpu,dtype=torch.int64).npu()
   golden=[];begin=0
   for i,c in enumerate(counts):
    if c:golden.append((xcpu[begin:begin+c].float()@wc[i].float().T)*scpu[begin:begin+c,None]*ws_cpu[i,None,:])
    begin+=c
   golden=torch.cat(golden)
   def call():return torch.ops.qwen36_mindieq_grouped.matmul(x,scale,packed,ws,ends)
   def serial():
    out=[];begin=0
    for i,c in enumerate(counts):
     if c:out.append(ext.OP(x[begin:begin+c],scale[begin:begin+c],packed[i],ws[i]))
     begin+=c
    return torch.cat(out)
   actual=call().cpu().float()
   max_abs=(actual-golden).abs().max().item()
   if not torch.isfinite(actual).all() or not torch.allclose(actual,golden,rtol=.002,atol=.002):
    row={'kind':'failure','case':label,'k':k,'n':n,'max_abs':max_abs,'bad_rows':((actual-golden).abs().amax(1)>.01).nonzero().flatten()[:64].tolist(),'finite':bool(torch.isfinite(actual).all())};f.write(json.dumps(row)+'\n');f.flush();raise AssertionError(row)
   torch.testing.assert_close(serial().cpu().float(),golden,rtol=.002,atol=.002)
   pending=[call() for _ in range(8)];torch.npu.synchronize()
   for o in pending:torch.testing.assert_close(o.cpu().float(),golden,rtol=.002,atol=.002)
   samples={'grouped':[],'serial':[]};functions={'grouped':call,'serial':serial}
   for fn in functions.values():
    for _ in range(5):fn()
   torch.npu.synchronize()
   for rep in range(a.repeats):
    for name in (['grouped','serial'] if rep%2==0 else ['serial','grouped']):
     begin=time.perf_counter();functions[name]();torch.npu.synchronize();samples[name].append((time.perf_counter()-begin)*1e3)
   graph=torch.npu.NPUGraph()
   with torch.npu.graph(graph):captured=call()
   graph.replay();torch.npu.synchronize();torch.testing.assert_close(captured.cpu().float(),golden,rtol=.002,atol=.002)
   # 更换设备端专家边界：保持总行数，重放不重新capture。
   reverse=list(reversed(counts));ends.copy_(torch.tensor(list(accumulate(reverse)),dtype=torch.int64).npu())
   alternate=[];begin=0
   for i,c in enumerate(reverse):
    if c:alternate.append((xcpu[begin:begin+c].float()@wc[i].float().T)*scpu[begin:begin+c,None]*ws_cpu[i,None,:])
    begin+=c
   graph.replay();torch.npu.synchronize();torch.testing.assert_close(captured.cpu().float(),torch.cat(alternate),rtol=.002,atol=.002)
   row={'kind':'measurement','case':label,'m':m,'e':e,'k':k,'n':n,'status':'PASS','max_abs':max_abs,'queued_outputs':8,'graph_dynamic_counts':'PASS','median_ms':{name:median(v) for name,v in samples.items()},'samples_ms':samples};f.write(json.dumps(row)+'\n');f.flush();print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
 f.write(json.dumps({'kind':'summary','status':'PASS'})+'\n')
