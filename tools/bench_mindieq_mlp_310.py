# SPDX-License-Identifier: Apache-2.0
"""真实权重MLP诊断（8/64/256专家）：固定分组，无router/共享专家/Attention/route combine。
计入激活量化、Gate/Up、SwiGLU、Down、Python分组调用；不外推模型吞吐。
"""
import argparse
from bisect import bisect_right
import importlib.util
import json
from itertools import accumulate
import time
from pathlib import Path
from statistics import median

import torch
import torch_npu
from safetensors import safe_open
from check_w4_repeatability_310 import load_bank
from check_w4_grouped_prefill_310 import signed_nibbles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--upstream-test',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--synchronize-producers',action='store_true')
    p.add_argument('--experts',type=int,choices=[8,64,256],default=8)
    p.add_argument('--rows',type=int,nargs='+',default=[8,128,512,1024,4096])
    p.add_argument('--retain-native-inputs',action='store_true',help='诊断：保留native pack/matmul参数，定位混合路径生命周期')
    p.add_argument('--backends',nargs='+',choices=['native_w4','mindieq_w4','w8','hybrid_128','hybrid_256'])
    p.add_argument('--graph',action='store_true',help='固定分组完整MLP Graph；3次激活更新与eager同后端输出比较')
    p.add_argument('--queued-repeats',type=int,default=0,help='完整MLP连续异步提交次数；统一同步后逐输出检查')
    p.add_argument('--activation-draw',type=int,default=0,help='跳过指定次4096x2048 CPU随机输入，重现原三分布测试的第3次输入用2')
    p.add_argument('--retain-intermediates',action='store_true',help='诊断张量生命周期：完整MLP同步前保留中间张量')
    p.add_argument('--validate-only',action='store_true',help='仅检查输出数值，故障诊断不执行warmup/计时')
    p.add_argument('--route-percentiles',type=float,nargs='+',default=[.05,.5,.95])
    p.add_argument('--diagnose-finite',action='store_true',help='逐投影同步检查，仅用于故障定位，不得用于测速')
    p.add_argument('--real-expert-weights',action='store_true',help='加载层0全部专家，避免重复8个专家带来的权重缓存偏差')
    p.add_argument('--hybrid',action='store_true',help='静态CPU分组诊断：小专家走native，128/256行以上走MindIEQ')
    p.add_argument('--route-trace',type=Path,help='真实256专家计数；完整真实权重需配合--real-expert-weights，不能代替端到端测试')
    a=p.parse_args()
    if a.route_trace and a.experts!=256:
        p.error('--route-trace requires --experts 256')
    torch.set_num_threads(4);torch.manual_seed(310)
    torch.ops.load_library(a.library)
    spec=importlib.util.spec_from_file_location('external_qmm',a.upstream_test)
    external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    weight_experts=a.experts if a.real_expert_weights else 8
    native=[load_bank(a.model,names,experts=weight_experts) for names in [('gate_proj','up_proj'),('down_proj',)]]
    other=[]
    for names in [('gate_proj','up_proj'),('down_proj',)]:
        banks=[]
        for expert in range(weight_experts):
            parts=[]
            for name in names:
                ts=[]
                for suffix in ['weight','weight_scale','weight_offset']:
                    key=f'model.language_model.layers.0.mlp.experts.{expert}.{name}.{suffix}'
                    with safe_open(a.model/index[key],framework='pt',device='cpu') as f:ts.append(f.get_tensor(key))
                parts.append(ts)
            codes,scales,offsets=[torch.cat(c) for c in zip(*parts)]
            assert torch.count_nonzero(offsets)==0
            assert torch.equal(scales,scales[:,:1].expand_as(scales))
            weight=signed_nibbles(codes).to(torch.int8);n,k=weight.shape
            packed=torch.from_numpy(external.pack_biased_nz_int32_vec(weight.T.contiguous().numpy(),k,n).copy()).npu()
            w8=torch_npu.npu_format_cast(weight.npu(),29).T
            banks.append((packed,w8,scales[:,0].float().npu()))
        other.append(banks)
    if a.experts!=8 and not a.real_expert_weights:
        native=[[t.repeat(a.experts//8,1,1) for t in bank] for bank in native]
        other=[bank*(a.experts//8) for bank in other]
    with a.out.open('x') as f:
        f.write(json.dumps({'kind':'config','scope':__doc__,'warmup':0 if a.validate_only else 5,'repeats':0 if a.validate_only else 20,'experts':a.experts,
                            'weight_scope':f'real layer0 expert0..{weight_experts-1}; repeated if weight_experts<experts',
                            'activations':'native fused pack vs torch_npu.npu_dynamic_quant',
                            'retain_native_inputs':a.retain_native_inputs,'graph':a.graph,'queued_repeats':a.queued_repeats,'activation_draw':a.activation_draw,'retain_intermediates':a.retain_intermediates,'validate_only':a.validate_only,'diagnose_finite':a.diagnose_finite,'synchronize_producers':a.synchronize_producers,'route_trace':str(a.route_trace) if a.route_trace else None,
                            'non_native_swiglu':'FP32 silu and multiply -> FP16 -> dynamic quant'})+'\n');f.flush()
        cases=[]
        if a.route_trace:
            traces=[json.loads(line) for line in a.route_trace.read_text().splitlines()]
            traces=[r for r in traces if r['chunk_tokens']==512]
            if not traces:
                raise ValueError('route trace has no full 512-token chunk')
            traces.sort(key=lambda r:max(r['counts']))
            for position in a.route_percentiles:
                if not 0<=position<=1:raise ValueError('route percentile must be in [0,1]')
                label=f'p{round(position*100):02d}'
                trace=traces[int((len(traces)-1)*position)]
                counts=trace['counts']
                assert len(counts)==a.experts and sum(counts)==4096
                cases.append((sum(counts),'real_occupancy_'+label,list(accumulate(counts))))
        else:
            for rows in a.rows:
                for routing in ['balanced','one_expert']:
                    ends=[rows*(i+1)//a.experts for i in range(a.experts)] if routing=='balanced' else [rows]*a.experts
                    cases.append((rows,routing,ends))
        for _ in range(a.activation_draw):torch.randn(4096,2048)
        for rows,routing,ends in cases:
            boundaries=torch.tensor(ends,dtype=torch.int64).npu()
            x=(torch.randn(rows,2048)*.2).half().npu()
            native_kept=[]
            def native_for(inputs,group_ends):
                pack=torch.ops.qwen36_int4.pack(inputs)
                if a.retain_native_inputs:native_kept.extend([inputs,*pack,group_ends])
                gate=torch.ops.qwen36_int4.matmul(*pack,*native[0],group_ends)
                if inputs.shape[0]<=128:
                    pack=torch.ops.qwen36_int4.swiglu_pack(gate)
                else:
                    g,u=gate.float().chunk(2,dim=-1)
                    pack=torch.ops.qwen36_int4.pack((torch.nn.functional.silu(g)*u).half())
                if a.retain_native_inputs:native_kept.extend([gate,*pack])
                return torch.ops.qwen36_int4.matmul(*pack,*native[1],group_ends)
            def native_mlp():
                return native_for(x,boundaries)
            def alternate_mlp(backend,selected=None):
                outputs=[];start=0;retained=[]
                # Each expert slice resident; counts fixed on host for this diagnostic.
                for expert,end in enumerate(ends):
                    if end==start:continue
                    if selected is not None and expert not in selected:
                        start=end;continue
                    q,s=torch_npu.npu_dynamic_quant(x[start:end])
                    s=s.float().contiguous()
                    q=torch_npu.npu_format_cast(q,2).contiguous()
                    w,w8,ws=other[0][expert]
                    if backend=='mindieq' and a.synchronize_producers:torch.npu.synchronize()
                    gate=external.OP(q,s,w,ws) if backend=='mindieq' else torch_npu.npu_quant_matmul(q,w8,ws,pertoken_scale=s,output_dtype=torch.float16)
                    if a.diagnose_finite:
                        assert torch.isfinite(gate).all().item(),('gate',backend,expert,end-start)
                    if a.retain_intermediates:retained.extend([q,s,gate])
                    g,u=gate.float().chunk(2,dim=-1)
                    hidden=(torch.nn.functional.silu(g)*u).half()
                    q,s=torch_npu.npu_dynamic_quant(hidden)
                    s=s.float().contiguous()
                    q=torch_npu.npu_format_cast(q,2).contiguous()
                    w,w8,ws=other[1][expert]
                    if backend=='mindieq' and a.synchronize_producers:torch.npu.synchronize()
                    out=external.OP(q,s,w,ws) if backend=='mindieq' else torch_npu.npu_quant_matmul(q,w8,ws,pertoken_scale=s,output_dtype=torch.float16)
                    if a.diagnose_finite:
                        assert torch.isfinite(out).all().item(),('down',backend,expert,end-start)
                    if a.retain_intermediates:retained.extend([hidden,q,s,out])
                    outputs.append(out);start=end
                result=torch.cat(outputs)
                if a.retain_intermediates:torch.npu.synchronize()
                return result
            funcs={'native_w4':native_mlp,'mindieq_w4':lambda:alternate_mlp('mindieq'),'w8':lambda:alternate_mlp('w8')}
            if a.hybrid:
                counts=[end-(ends[i-1] if i else 0) for i,end in enumerate(ends)]
                for threshold in [128,256]:
                    large={i for i,count in enumerate(counts) if count>=threshold}
                    small_indices=[];large_indices=[];small_counts=[];begin=0
                    for expert,end in enumerate(ends):
                        (large_indices if expert in large else small_indices).extend(range(begin,end))
                        small_counts.append(0 if expert in large else end-begin);begin=end
                    small_ids=torch.tensor(small_indices,dtype=torch.int64).npu()
                    small_ends=torch.tensor(list(accumulate(small_counts)),dtype=torch.int64).npu()
                    inverse=torch.argsort(torch.tensor(small_indices+large_indices,dtype=torch.int64)).npu()
                    def hybrid(large=large,small_ids=small_ids,small_ends=small_ends,inverse=inverse):
                        outputs=[]
                        if small_ids.numel():outputs.append(native_for(x.index_select(0,small_ids),small_ends))
                        if large:outputs.append(alternate_mlp('mindieq',large))
                        return torch.cat(outputs).index_select(0,inverse)
                    funcs[f'hybrid_{threshold}']=hybrid
            if a.backends:funcs={name:funcs[name] for name in a.backends}
            reference=native_mlp().cpu().float();errors={}
            for name,fn in funcs.items():
                output=fn().cpu().float()
                errors[name]={'max_abs':(output-reference).abs().max().item(),
                              'relative_l2':((output-reference).norm()/reference.norm().clamp_min(1e-12)).item(),
                              'finite':bool(torch.isfinite(output).all())}
                if not errors[name]['finite']:
                    bad_rows=(~torch.isfinite(output).all(dim=-1)).nonzero().flatten().tolist()
                    histogram={}
                    for bad_row in bad_rows:
                        expert=bisect_right(ends,bad_row)
                        histogram[expert]=histogram.get(expert,0)+1
                    failure={'kind':'failure','backend':name,'routing':routing,'rows':rows,'errors':errors[name],
                             'bad_expert_rows':histogram,'bad_rows_first64':bad_rows[:64],'group_ends':ends}
                    f.write(json.dumps(failure)+'\n');f.flush()
                    raise AssertionError(failure)
                # Independent per-token quantizers can differ at rounding boundaries.
                assert errors[name]['relative_l2']<.03,errors[name]
                if a.queued_repeats:
                    pending=[fn() for _ in range(a.queued_repeats)]
                    torch.npu.synchronize()
                    queue_errors=[];queue_replay_error=[]
                    for queued in pending:
                        queued=queued.cpu().float()
                        finite=bool(torch.isfinite(queued).all())
                        relative=((queued-reference).norm()/reference.norm().clamp_min(1e-12)).item()
                        assert finite and relative<.03,(routing,name,'queued',finite,relative)
                        assert torch.equal(queued,output),(routing,name,'queued changed output')
                        queue_errors.append(relative)
                        queue_replay_error.append((queued-output).abs().max().item())
                    errors[name]['queued_relative_l2']=queue_errors
                    errors[name]['queued_max_abs_vs_first']=queue_replay_error
                if not a.validate_only:
                    for _ in range(5):fn()
            graph_checks={}
            if a.graph:
                saved=x.clone();graphs={};captured={}
                for name,fn in funcs.items():
                    for _ in range(3):fn()
                    torch.npu.synchronize()
                    graph=torch.npu.NPUGraph()
                    with torch.npu.graph(graph):captured[name]=fn()
                    graphs[name]=graph
                for variant in range(3):
                    x.copy_((torch.randn(rows,2048)*(.1+.1*variant)).half().npu())
                    torch.npu.synchronize()
                    for name,fn in funcs.items():
                        expected=fn().cpu().float()
                        graphs[name].replay();torch.npu.synchronize()
                        actual=captured[name].cpu().float()
                        torch.testing.assert_close(actual,expected,rtol=.002,atol=.0002)
                        graph_checks.setdefault(name,[]).append({'variant':variant,'max_abs':(actual-expected).abs().max().item()})
                x.copy_(saved);torch.npu.synchronize()
            if a.validate_only:
                row={'kind':'validation','routing':routing,'rows':rows,'status':'PASS','timing_valid':False,'errors_vs_native':errors,'graph_checks':graph_checks}
                f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
                continue
            torch.npu.synchronize();samples={name:[] for name in funcs}
            for repeat in range(20):
                order=list(funcs);order=order[repeat%len(order):]+order[:repeat%len(order)]
                for name in order:
                    begin=time.perf_counter();funcs[name]();torch.npu.synchronize()
                    samples[name].append((time.perf_counter()-begin)*1000)
            row={'kind':'measurement','rows':rows,'routing':routing,'group_ends':ends,
                 'status':'PASS','errors_vs_native':errors,'graph_checks':graph_checks,'latency_ms':samples,
                 'median_ms':{name:median(v) for name,v in samples.items()},
                 'timing_valid':not (a.diagnose_finite or a.retain_intermediates or a.synchronize_producers or a.retain_native_inputs),'scope':'synchronized full MLP fixed grouping; no router/combine'}
            f.write(json.dumps(row)+'\n');f.flush()
            print(json.dumps({k:v for k,v in row.items() if k!='latency_ms'}),flush=True)
        f.write(json.dumps({'kind':'summary','status':'PASS','cases':len(cases)})+'\n')


if __name__=='__main__':main()
