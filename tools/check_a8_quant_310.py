# SPDX-License-Identifier: Apache-2.0
"""独立A8量化门禁：FP32 CPU参考、padding、异步提交、Graph及公平准备耗时。"""
import argparse,json,time
from pathlib import Path
from statistics import median
import torch
import torch_npu

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(310);torch.ops.load_library(a.library)
op=torch.ops.qwen36_mindieq_grouped.quantize

def reference(x):
    maximum=x.float().abs().amax(1)
    scale=torch.where(maximum==0,torch.ones_like(maximum),maximum/127)
    q=(x.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
    return torch.cat((q,torch.zeros(16,x.shape[1],dtype=torch.int8))),torch.cat((scale,torch.zeros(16)))

def dynamic(x):
    q,s=torch_npu.npu_dynamic_quant(x)
    q=torch_npu.npu_format_cast(q,2).contiguous()
    return torch.cat((q,torch.zeros(16,x.shape[1],dtype=torch.int8,device=x.device))),torch.cat((s.float(),torch.zeros(16,device=x.device)))

def check(value,golden):
    torch.npu.synchronize()
    q,s=[t.cpu() for t in value];gq,gs=golden
    if not torch.equal(q,gq) or not torch.allclose(s,gs,rtol=1e-6,atol=0):
        raise RuntimeError(f'Quantization failed: codes={int((q!=gq).sum())},scale_max={float((s-gs).abs().max())}')

with a.out.open('x') as f:
    def emit(row):f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
    emit({'kind':'config','scope':__doc__,'warmup':5,'repeats':20,'timing':'host dispatch+synchronize, both backends include 16 padding rows','seed':310})
    for k in (512,2048):
        for m in (1,2,4,8,17,128,512,1024,4096):
            x=(torch.randn(m,k)*.2).half();x[0]*=.0001
            if m>1:x[1].zero_()
            if m>2:x[2].fill_(-1)
            if m>3:x[3]=torch.linspace(-1,1,k).half()
            golden=reference(x);device=x.npu();check(op(device),golden)
            queued=[op(device) for _ in range(8)]
            for out in queued:check(out,golden)
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):captured=op(device)
            # 同shape更新输入，证明Graph没有冻结scale及输入。
            replacement=x.clone();replacement[0].fill_(.03125)
            device.copy_(replacement.npu());graph.replay();check(captured,reference(replacement))
            device.copy_(x.npu());torch.npu.synchronize()
            ops={'custom_fp32_arithmetic':op,'dynamic_fp16':dynamic}
            for fn in ops.values():
                for _ in range(5):fn(device)
            torch.npu.synchronize();samples={name:[] for name in ops}
            for repeat in range(20):
                for name in list(ops) if repeat%2==0 else list(reversed(ops)):
                    start=time.perf_counter();out=ops[name](device);torch.npu.synchronize();samples[name].append((time.perf_counter()-start)*1e6)
            emit({'kind':'case','m':m,'k':k,'status':'PASS','queued_outputs':8,'graph_changed_input':'PASS','median_us':{n:median(v) for n,v in samples.items()},'samples_us':samples})
    emit({'kind':'summary','status':'PASS','cases':18})
