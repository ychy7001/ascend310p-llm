# SPDX-License-Identifier: Apache-2.0
"""诊断：真实expert0、相同整数/scale的三后端单矩阵eager/Graph对照。
激活为合成FP16，由当前设备pack生成共同A8；不含pack/路由/SwiGLU耗时。
Graph固定shape/address，回放前原地更新四组输入，不证明vLLM动态图兼容。
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from statistics import median

import torch
import torch_npu
from safetensors import safe_open
from check_w4_repeatability_310 import native_weight, native_metadata
from check_w4_grouped_prefill_310 import signed_nibbles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--upstream-test', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.ops.load_library(args.library)
    spec = importlib.util.spec_from_file_location('external_qmm', args.upstream_test)
    external = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external)
    index = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    with args.out.open('x') as file:
        file.write(json.dumps({'kind':'config','scope':__doc__,'warmup':5,'repeats':args.repeats,
                              'seed':20261013,'expert':0,'layer':0,
                              'model_index_sha256':hashlib.sha256((args.model/'model.safetensors.index.json').read_bytes()).hexdigest()})+'\n')
        for projection, names in [('gate_up',('gate_proj','up_proj')),('down',('down_proj',))]:
            parts=[]
            for name in names:
                tensors=[]
                for suffix in ('weight','weight_scale','weight_offset'):
                    key=f'model.language_model.layers.0.mlp.experts.0.{name}.{suffix}'
                    with safe_open(args.model/index[key],framework='pt',device='cpu') as source:
                        tensors.append(source.get_tensor(key))
                parts.append(tensors)
            codes, scales, offsets = [torch.cat(column) for column in zip(*parts)]
            assert torch.count_nonzero(offsets)==0, 'Requires symmetric checkpoint'
            assert torch.equal(scales,scales[:, :1].expand_as(scales)), 'Requires per-channel scale'
            weight=signed_nibbles(codes).to(torch.int8)
            n,k=weight.shape
            native,sums=native_weight(codes)
            bank=[native.unsqueeze(0).npu(),native_metadata(scales).unsqueeze(0).npu(),
                  native_metadata(offsets).unsqueeze(0).npu(),sums.unsqueeze(0).npu()]
            packed=torch.from_numpy(external.pack_biased_nz_int32_vec(weight.T.contiguous().numpy(),k,n).copy()).npu()
            w8=torch_npu.npu_format_cast(weight.npu(),29).T
            ws=scales[:,0].float().npu()
            for m in (1,2,4,8,128,512,1024,4096):
                variants=[]
                for variant in range(4):
                    x=(torch.randn(m,k)*.2).half().npu()
                    p=torch.ops.qwen36_int4.pack(x)
                    lo,hi,ps,_=[t.cpu() for t in p]
                    assert torch.equal(ps,ps[:,:1,:1].expand_as(ps)), 'Requires per-token activation scale'
                    quant=(signed_nibbles(lo)+16*signed_nibbles(hi)+8).to(torch.int8)
                    actscale=ps[:,0,0].float()
                    golden=(quant.float()@weight.float().T)*actscale[:,None]*scales[:,0].float()[None,:]
                    variants.append((p,quant.npu(),actscale.npu(),golden))
                current,tx,ts,_=variants[0]
                current=tuple(t.clone() for t in current);tx=tx.clone();ts=ts.clone()
                ends=torch.tensor([m],dtype=torch.int64).npu()
                funcs={'native_w4':lambda:torch.ops.qwen36_int4.matmul(*current,*bank,ends),
                       'mindieq_w4':lambda:external.OP(tx,ts,packed,ws),
                       'w8':lambda:torch_npu.npu_quant_matmul(tx,w8,ws,pertoken_scale=ts,output_dtype=torch.float16)}
                def update(v):
                    p,x,s,_=v
                    for a,b in zip(current,p):a.copy_(b)
                    tx.copy_(x);ts.copy_(s)
                    torch.npu.synchronize()
                errors={}
                for name,fn in funcs.items():
                    for v in variants:
                        update(v);actual=fn().cpu().float()
                        torch.testing.assert_close(actual,v[3],rtol=.002,atol=.002)
                        errors[name]=max(errors.get(name,0),(actual-v[3]).abs().max().item())
                update(variants[0])
                samples={name:[] for name in funcs}
                for fn in funcs.values():
                    for _ in range(5):fn()
                torch.npu.synchronize()
                for repeat in range(args.repeats):
                    order=list(funcs);order=order[repeat%3:]+order[:repeat%3]
                    for name in order:
                        begin=time.perf_counter();funcs[name]();torch.npu.synchronize()
                        samples[name].append((time.perf_counter()-begin)*1e6)
                graph={};handles={};outputs={};graph_errors={}
                try:
                    for name,fn in funcs.items():
                        handles[name]=torch.npu.NPUGraph()
                        with torch.npu.graph(handles[name]):outputs[name]=fn()
                        for variant,v in enumerate(variants):
                            update(v);handles[name].replay();torch.npu.synchronize()
                            try:
                                torch.testing.assert_close(outputs[name].cpu().float(),v[3],rtol=.002,atol=.002)
                            except AssertionError as error:
                                graph_errors[name]={'variant':variant,'error':str(error),'max_abs':(outputs[name].cpu().float()-v[3]).abs().max().item()}
                                break
                        if name in graph_errors:
                            print(json.dumps({'graph_check_failed':name,**graph_errors[name]}),flush=True)
                            continue
                        graph[name]=[]
                        for _ in range(5):handles[name].replay()
                        torch.npu.synchronize()
                    for repeat in range(args.repeats):
                        order=list(graph);order=order[repeat%3:]+order[:repeat%3]
                        for name in order:
                            begin=time.perf_counter();handles[name].replay();torch.npu.synchronize()
                            graph[name].append((time.perf_counter()-begin)*1e6)
                finally:
                    handles.clear()
                row={'kind':'measurement','projection':projection,'m':m,'k':k,'n':n,
                     'status':'PASS' if not graph_errors else 'GRAPH_CHECK_FAILED','graph_errors':graph_errors,'max_abs':errors,'dynamic_input_variants':4,
                     'eager_us':samples,'graph_us':graph,
                     'eager_median_us':{name:median(v) for name,v in samples.items()},
                     'graph_median_us':{name:median(v) for name,v in graph.items()},
                     'weight_bytes':{'native_w4':codes.numel(),'mindieq_w4':packed.numel()*4,'w8':w8.numel()},
                     'timing_scope':'host dispatch/replay + synchronize; pack/input update excluded'}
                file.write(json.dumps(row)+'\n');file.flush()
                print(json.dumps({key:value for key,value in row.items() if key not in ('eager_us','graph_us')}),flush=True)
        file.write(json.dumps({'kind':'summary','status':'COMPLETE','cases':16})+'\n')


if __name__=='__main__':main()
