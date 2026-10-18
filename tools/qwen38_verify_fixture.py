"""Real checkpoint fixture gates. CPU PLE + optional real 310P W4 kernel.
Not a full model accuracy/throughput benchmark; activations and routes are synthetic.
"""
import argparse, hashlib, importlib, json, resource, sys, time, types
from pathlib import Path
import numpy as np
import torch
# SPDX-License-Identifier: Apache-2.0
from safetensors.torch import load_file

def module_path(name,path):
    mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--library');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(31038)
    manifest=json.loads((a.fixture/'manifest.json').read_text());results={'revision':manifest['revision'],'scope':'real experts 0/256, layer 0; synthetic activations/routes; selected real PLE rows only'}
    if a.source:
        for name,relative in [('vllm_ascend','vllm_ascend'),('vllm_ascend.models','vllm_ascend/models'),('vllm_ascend.models.qwen4_exp','vllm_ascend/models/qwen4_exp'),('vllm_ascend.observability','vllm_ascend/observability')]:
            module_path(name,a.source/relative)
        cls=importlib.import_module('vllm_ascend.models.qwen4_exp.ngram_embedding').AscendPLELazyShardEmbeddingMethod
        before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ple=cls(manifest['ple_shard_rows']*128,manifest['ple_dim'],checkpoint_dir=a.fixture,index_filename='model.safetensors.index.json')
        ids=torch.tensor([r['id'] for r in manifest['ple_rows']]);t=time.perf_counter();actual=ple.gather_rows(ids);elapsed=time.perf_counter()-t
        for row,expected in zip(actual,manifest['ple_rows']):
            assert hashlib.sha256(row.numpy().tobytes()).hexdigest()==expected['sha256']
        # Repeat boundary/duplicate accesses using the original implementation.
        for _ in range(20): torch.testing.assert_close(ple.gather_rows(ids.flip(0)),actual.flip(0),rtol=0,atol=0)
        results['ple']={'status':'PASS','real_rows':len(ids),'cold_fixture_gather_ms':elapsed*1000,'maxrss_delta_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-before,'logical_table_bytes':manifest['ple_shard_rows']*128*manifest['ple_dim']*2,'fixture_only':True,'note':'Sparse fixture; cold latency is NOT representative of full-table random I/O. mmap physical_bytes=0 does not mean OS page-cache memory is zero.'}
        ple.close()
        from vllm_ascend.models.qwen4_exp.qsa_head_sharding import qsa_head_shard
        from vllm_ascend.models.qwen4_exp.ops.qsa_attention import qsa_sparse_gqa_attention
        q=torch.randn(8,24,256);k=torch.randn(64,2,256);v=torch.randn_like(k);gate=torch.randn_like(q)
        selected=torch.arange(0,64,2).repeat(8,1);counts=torch.full((8,),32)
        whole=qsa_sparse_gqa_attention(q,k,v,gate,selected,counts,2,accum_dtype=torch.float32)
        parts=[]
        for rank in range(2):
            sh=qsa_head_shard(24,2,256,rank,2);qr=slice(sh.query_start,sh.query_start+sh.num_query_heads);kr=slice(sh.kv_start,sh.kv_start+sh.num_kv_heads)
            parts.append(qsa_sparse_gqa_attention(q[:,qr],k[:,kr],v[:,kr],gate[:,qr],selected,counts,1,accum_dtype=torch.float32))
        combined=torch.cat(parts,dim=1);torch.testing.assert_close(combined,whole,rtol=1e-5,atol=1e-6)
        results['qsa_tp2_host_math']={'status':'PASS','query_heads_per_rank':12,'kv_heads_per_rank':1,'head_dim':256,'max_abs':float((combined-whole).abs().max()),'scope':'CPU sparse-attention head slicing only; no HCCL, projections, NPU kernels, or whole model'}
    if a.library:
        import torch_npu
        torch.ops.load_library(a.library);torch.npu.set_device(0)
        tensors=load_file(a.fixture/'experts.safetensors');rows=[]
        def metadata(t):return t.reshape(t.shape[0]//16,16,-1).transpose(1,2).contiguous().view_as(t)
        def get(e,proj,kind):return tensors[f'model.language_model.layers.0.mlp.experts.{e}.{proj}.{kind}']
        for proj in ('gate_proj','up_proj','down_proj'):
            codes=[];scales=[];offsets=[];sums=[];weights=[]
            for e in (0,256):
                w=get(e,proj,'weight');s=get(e,proj,'weight_scale');z=get(e,proj,'weight_offset');n,k2=w.shape
                nib=torch.stack((w&15,(w>>4)&15),-1).flatten(-2);q=torch.where(nib>=8,nib-16,nib).float()
                weights.append(((q.reshape(n,-1,128)-z.float().unsqueeze(-1))*s.float().unsqueeze(-1)).reshape(n,-1))
                codes.append(w.reshape(n//16,16,k2//64,2,32).permute(0,2,3,1,4).contiguous().view_as(w))
                scales.append(metadata(s));offsets.append(metadata(z.half()));sums.append(metadata(q.reshape(n,-1,128).sum(-1).half()))
            banks=[torch.stack(v).npu() for v in (codes,scales,offsets,sums)]
            for m in (1,2,4,8,128,512):
                x=(torch.randn(m,k2*2)*.2).half();xn=x.npu()
                if m<=8:
                    # two routed outputs per token, exercises both TP ownership ranges.
                    routing=torch.tensor([0,1]*m,dtype=torch.int32).npu();source=x.repeat_interleave(2,0);expert_ids=[0,1]*m
                else:
                    routing=torch.tensor([m//2,m],dtype=torch.int64).npu();source=x;expert_ids=[0]*(m//2)+[1]*(m-m//2)
                def operation():return torch.ops.qwen36_int4.matmul(*torch.ops.qwen36_int4.pack(xn),*banks,routing)
                if m == 1:
                    low,high,ascale,asum=torch.ops.qwen36_int4.pack(xn)
                    def unpack(t):
                        t=t.cpu();u=torch.stack((t&15,(t>>4)&15),-1).flatten(-2)
                        return torch.where(u>=8,u-16,u).float()
                    decoded=unpack(low)+16*unpack(high)+8
                    gg=x.float().reshape(m,-1,128);sc=gg.abs().amax(-1)/127
                    qq=(gg/sc.unsqueeze(-1)).round().clamp(-127,127).flatten(1)
                    print(json.dumps({'diagnostic':proj,'activation_quant_maxdiff':float((decoded-qq).abs().max()),'activation_scale_maxdiff':float((ascale.cpu()[...,0]-sc).abs().max()),'activation_sum_maxdiff':float((asum.cpu()[...,0]-qq.reshape(m,-1,128).sum(-1)).abs().max())}),flush=True)
                for _ in range(5):operation()
                samples=[]
                for sample in range(3):
                    torch.npu.synchronize();t=time.perf_counter()
                    for _ in range(20): y=operation()
                    torch.npu.synchronize();samples.append((time.perf_counter()-t)*1000/20)
                ms=sorted(samples)[1];y=y.cpu().float()
                g=source.float().reshape(-1,k2*2//128,128);scale=g.abs().amax(-1,keepdim=True)/127
                scale=torch.where(scale==0,torch.ones_like(scale),scale);quant=(g/scale).round().clamp(-127,127)
                aq=(quant*scale).flatten(1);ref=torch.stack([aq[i]@weights[e].T for i,e in enumerate(expert_ids)])
                fp=torch.stack([source[i].float()@weights[e].T for i,e in enumerate(expert_ids)])
                err=(y-ref).norm()/ref.norm();activation_err=(y-fp).norm()/fp.norm()
                assert torch.isfinite(y).all(),(proj,m,'nonfinite')
                rows.append({'projection':proj,'input_tokens':m,'output_routes':len(expert_ids),'mode':'routed' if m<=8 else 'grouped','shape_N_K':[n,k2*2],'latency_pack_plus_matmul_ms':ms,'latency_samples_ms':samples,'rel_l2_vs_w4a8_float_reference':float(err),'rel_l2_vs_w4a16_float_reference':float(activation_err),'max_abs_vs_w4a8_reference':float((y-ref).abs().max()),'cosine_vs_reference':float(torch.nn.functional.cosine_similarity(y.flatten(),ref.flatten(),dim=0))})
                print(json.dumps(rows[-1]),flush=True)
        results['npu']={'status':'PASS' if all(r['rel_l2_vs_w4a8_float_reference'] < .005 for r in rows) else 'FAIL','device':torch.npu.get_device_name(0),'rows':rows,'note':'Standalone W4 operators only. No full serving runtime or BF16 source-weight accuracy claim. latency=host dispatch of pack+matmul and batch synchronization, not isolated kernel time.','library_sha256':hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),'warmup':5,'timing_samples':3,'operations_per_sample':20}
    a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results if not a.library else {'npu':results['npu']['status'],'cases':len(rows)}))
    if a.library and results['npu']['status'] != 'PASS':sys.exit(1)
if __name__=='__main__':main()
