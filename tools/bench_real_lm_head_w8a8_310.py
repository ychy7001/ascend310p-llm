# SPDX-License-Identifier: Apache-2.0
"""Real Qwen3.6 LM-head FP16 versus 310P W8A8; standalone, no model service."""
import argparse,json,time
from pathlib import Path
from statistics import median
import torch, torch_npu
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
idx=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']['lm_head.weight']
with safe_open(str(a.model/idx),framework='pt',device='cpu') as f: w=f.get_tensor('lm_head.weight').contiguous()
assert w.shape==(248320,2048) and w.dtype in (torch.bfloat16,torch.float16)
# Per-output symmetric W8: input x [M,K], weight op expects [K,N] NZ, scale [N].
wf=w.float();# 310P QuantMatmulWeightNz rejects N>65535; emulate four vocab shards.
parts=[]
for lo in range(0,w.shape[0],62080):
  wh=w[lo:lo+62080].contiguous(); whf=wh.float(); sh=whf.abs().amax(1).clamp_min(1e-8)/127
  qh=torch.round(whf/sh[:,None]).clamp(-127,127).to(torch.int8)
  parts.append((wh.npu(),torch_npu.npu_format_cast(qh.contiguous().npu(),29).transpose(0,1),sh.float().npu()))
assert len(parts)==4 and all(x[1].shape[-1] <= 65535 for x in parts)
rows=[]
def bench(fn):
 for _ in range(5): fn()
 torch.npu.synchronize();v=[]
 for _ in range(30):
  t=time.perf_counter();y=fn();torch.npu.synchronize();v.append((time.perf_counter()-t)*1e6)
 return median(v),v
for m in (1,2,3,8):
 x=torch.randn(m,2048,dtype=torch.float16);xn=x.npu(); qout=torch.empty((m,w.shape[0]),device='npu',dtype=torch.float16)
 def fpfn():return torch.cat([torch.matmul(xn,ph.t()) for ph,_,_ in parts],dim=1)
 def qfn():
  x8,xs=torch_npu.npu_dynamic_quant(xn); x8=x8.squeeze(1) if x8.dim()==3 else x8; xs=xs.squeeze(1) if xs.dim()==2 else xs
  for i,(_,pw,ps) in enumerate(parts):
   lo=i*62080; qout[:,lo:lo+pw.shape[1]].copy_(torch_npu.npu_quant_matmul(x8,pw,ps,pertoken_scale=xs,output_dtype=torch.float16))
  return qout
 yf=fpfn();yq=qfn();torch.npu.synchronize();yf=yf.cpu().float();yq=yq.cpu().float()
 assert torch.isfinite(yf).all() and torch.isfinite(yq).all(), {'fp16_finite':bool(torch.isfinite(yf).all()),'w8a8_finite':bool(torch.isfinite(yq).all())}
 e=(yq-yf);fm,fs=bench(fpfn);qm,qs=bench(qfn)
 graph_us=None; graph_status='not-run'
 if m==1:
  try:
   qfn(); torch.npu.synchronize(); g=torch.npu.NPUGraph()
   with torch.npu.graph(g): graph_out=qfn()
   for _ in range(5): g.replay()
   torch.npu.synchronize(); graph_error=float((qout.cpu().float()-yq).abs().max()); gs=[]
   for _ in range(30):
    t=time.perf_counter();g.replay();torch.npu.synchronize();gs.append((time.perf_counter()-t)*1e6)
   graph_us=median(gs); graph_status='PASS'
  except Exception as ex:
   graph_status=type(ex).__name__+': '+str(ex)[:240]
 rows.append({'m':m,'status':'PASS','fp16_us':fm,'w8a8_us':qm,'fp16_baseline':'ND torch.matmul per 62080 shard; not comparable to production FP16 NZ','max_abs':e.abs().max().item(),'relative_l2':(e.norm()/yf.norm().clamp_min(1e-8)).item(),'fp16_weight_bytes':w.numel()*w.element_size(),'w8_weight_bytes':sum(x[1].numel() for x in parts),'scale_bytes':sum(x[2].numel()*x[2].element_size() for x in parts),'timing':'host dispatch+launch+synchronize warmup5 repeats30','graph_us':graph_us,'graph_status':graph_status,'graph_max_abs_vs_eager':(graph_error if m==1 and graph_status=='PASS' else None),'topk_overlap':({k:float(torch.isin(torch.topk(yf[0],k).indices,torch.topk(yq[0],k).indices).float().mean()) for k in (1,5,10,50)} if m==1 else None)})
with a.out.open('w') as f:
 for r in rows:f.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
