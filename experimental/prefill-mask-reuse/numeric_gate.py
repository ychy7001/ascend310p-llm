import json
from types import SimpleNamespace
from eager_mask_reuse import one_mask, scoped_forward
import torch
import torch_npu
import ast
from pathlib import Path

# Execute exact function bodies from the pinned image, without importing the
# serving package (standalone import enters a worker/plugin circular import).
root=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
def load_fn(file,name,class_name=None):
    nodes=ast.parse((root/file).read_text()).body
    if class_name:
        nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    mod=ast.Module(body=[node],type_ignores=[])
    exec(compile(mod,str(root/file),'exec',flags=__import__('__future__').annotations.compiler_flag),globals())
    return globals()[name]
ACL_FORMAT_FRACTAL_NZ=29
load_fn('utils.py','nd_to_nz_spec')
mask_file='_310p/attention/attention_mask.py'
load_fn(mask_file,'is_compressed_mask_supported')
mask_fn=load_fn(mask_file,'get_splitfuse_mask','AttentionMaskBuilder310')

block, heads, kv, dim = 64, 8, 1, 256
torch.set_num_threads(1)
torch.manual_seed(314159)
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format=True
assert not is_compressed_mask_supported()
layouts = {
    'cold_long': ([1920], [30720]),
    'prefix_hit_shape': ([1920], [13440]),
    'mixed': ([512,384,1024], [12544,30720,1024]),
    'mixed_reordered': ([384,1024,512], [30720,1024,12544]),
}
capacity = sum(((c+block-1)//block)*block for c in layouts['mixed'][1]); slots = torch.arange(capacity, dtype=torch.int32).npu()
shape = (capacity//block, kv*dim//16, block, 16)
kcache = torch_npu.empty_with_format(size=shape,device='npu',dtype=torch.float16,acl_format=29)
vcache = torch_npu.empty_with_format(size=shape,device='npu',dtype=torch.float16,acl_format=29)
k_cpu=torch.randn((capacity,kv,dim),dtype=torch.float32).half(); v_cpu=torch.randn_like(k_cpu)
k=k_cpu.npu(); v=v_cpu.npu()
for idx in range(0,capacity,3840):
    s=slice(idx, idx+3840)
    torch_npu._npu_reshape_and_cache(key=k[s],value=v[s],key_cache=kcache,value_cache=vcache,slot_indices=slots[s])
torch.npu.synchronize()

@scoped_forward
def check(name, qlens, contexts):
    starts = [0]
    for q in qlens:
        starts.append(starts[-1]+q)
    host_q=torch.tensor(qlens,dtype=torch.int32); ctx=torch.tensor(contexts,dtype=torch.int32).npu()
    meta=SimpleNamespace(query_start_loc=torch.tensor(starts,dtype=torch.int32).npu(),seq_lens=ctx)
    table=torch.zeros((len(contexts),(max(contexts)+block-1)//block),dtype=torch.int32)
    offsets=[]; page=0
    for i, c in enumerate(contexts):
        offsets.append(page*block)
        pages=(c+block-1)//block
        table[i,:pages]=torch.arange(page,page+pages)
        page+=pages
    table=table.npu(); mask_key=(ctx.device, torch.float16, 'legacy_splitfuse_nz', tuple(starts), tuple(contexts))
    def build(): return mask_fn(None,meta,ctx.device)
    mask=one_mask(mask_key,build)
    errors=[]
    for _ in range(2):
        q_cpu=torch.randn((starts[-1],heads,dim),dtype=torch.float32).half(); q=q_cpu.npu()
        out=torch.empty_like(q)
        assert one_mask(mask_key,build) is mask
        def pa(m):
            torch_npu._npu_paged_attention_splitfuse(query=q,key_cache=kcache,value_cache=vcache,mask=m,
                block_table=table,seq_len=host_q,context_lens=ctx,num_kv_heads=kv,num_heads=heads,
                scale_value=dim**-.5,out=out)
            return out.cpu()
        actual=pa(mask); fresh=pa(build())
        assert torch.equal(actual, fresh) and torch.isfinite(actual).all(), name
        for i,(n,c,offset) in enumerate(zip(qlens,contexts,offsets)):
            local=torch.tensor([0,n//2,n-1]); ix=starts[i]+local
            scores=(q_cpu[ix].float() @ k_cpu[offset:offset+c,0,:].float().T)*(dim**-.5)
            scores.masked_fill_(torch.arange(c)[None,None,:] > (c-n+local)[:,None,None], -float('inf'))
            ref=scores.softmax(-1) @ v_cpu[offset:offset+c,0,:].float()
            errors.append((actual[ix].float()-ref).abs().max().item())
    assert max(errors)<0.005,(name,errors)
    print(json.dumps(dict(layout=name,query=qlens,context=contexts,full_fp16_bitwise_equal=True,
        query_trials=2,max_abs_error_vs_cpu_fp32=max(errors))),flush=True)

for name,(queries,contexts) in layouts.items():
    check(name,queries,contexts)
