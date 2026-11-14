# SPDX-License-Identifier: Apache-2.0
"""Compare the pinned upstream and proposed mask builders on legacy SplitFuse."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu


def load(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(selected) == len(names), (path, names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), namespace)


base = '/vllm-workspace/vllm-ascend/vllm_ascend/'
ns = dict(torch=torch, torch_npu=torch_npu, AscendMetadata=SimpleNamespace, ACL_FORMAT_FRACTAL_NZ=29,
          COMPRESSED_MASK_SEQ_LEN=2048, PAGED_ATTENTION_COMPRESSED_MASK_VALUE=-10000)
load(base+'utils.py', {'nd_to_nz_spec'}, ns)
for variant in ('upstream', 'candidate'):
    load(f'/workbench/{variant}.py', {'AttentionMaskBuilder310', 'is_compressed_mask_supported'}, ns)
    ns[variant] = ns['AttentionMaskBuilder310']
old, new = ns['upstream'], ns['candidate']
assert not ns['is_compressed_mask_supported']()
torch.set_num_threads(1)
torch.manual_seed(20261114)
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = True
device = torch.device('npu:0')
block, heads, dim = 64, 8, 256

for queries, contexts, compare_old in (([64],[127],True), ([17,15,1],[129,65,33],True),
                                       ([1,17,15],[33,129,65],True), ([3,1920,3],[14366,32774,1029],False)):
    starts = [0]
    for q in queries:
        starts.append(starts[-1]+q)
    pages = [(c+block-1)//block for c in contexts]
    total = sum(pages)*block
    query = torch.randn(starts[-1],heads,dim).half()
    key = torch.randn(total,1,dim).half()
    value = torch.randn_like(key)
    kv_shape = (sum(pages),dim//16,block,16)
    kc, vc = [torch_npu.empty_with_format(size=kv_shape, device=device, dtype=torch.float16, acl_format=29) for _ in range(2)]
    slots = torch.arange(total,dtype=torch.int32).to(device)
    for offset in range(0,total,3840):
        span = slice(offset,offset+3840)
        torch_npu._npu_reshape_and_cache(key=key[span].to(device),value=value[span].to(device),
                                        key_cache=kc,value_cache=vc,slot_indices=slots[span])
    table = torch.zeros(len(contexts),max(pages),dtype=torch.int32)
    page = 0
    for i,n in enumerate(pages):
        table[i,:n] = torch.arange(page,page+n)
        page += n
    qn, table = query.to(device), table.to(device)
    qlens = torch.tensor(queries,dtype=torch.int32)
    context = torch.tensor(contexts,dtype=torch.int32).to(device)
    meta = SimpleNamespace(query_start_loc=torch.tensor(starts,dtype=torch.int32).to(device),seq_lens=context)
    out = torch.empty_like(qn)
    def run(mask):
        torch_npu._npu_paged_attention_splitfuse(query=qn,key_cache=kc,value_cache=vc,mask=mask,
            block_table=table,seq_len=qlens,context_lens=context,num_kv_heads=1,num_heads=heads,
            scale_value=dim**-.5,out=out)
        return out.cpu()
    active = new.get_splitfuse_mask(meta,device,active_width=True)
    actual = run(active)
    if compare_old:
        old.max_seqlen = new.max_seqlen = 512
        assert torch.equal(actual.view(torch.int16),run(old.get_splitfuse_mask(meta,device)).view(torch.int16))
        assert torch.equal(actual.view(torch.int16),run(new.get_splitfuse_mask(meta,device)).view(torch.int16))
    offset, errors = 0, []
    for i,(q,c,n) in enumerate(zip(queries,contexts,pages,strict=True)):
        local = torch.tensor(sorted({0,q//2,q-1})); ix = starts[i]+local
        scores = (query[ix].float() @ key[offset:offset+c,0].float().T)*(dim**-.5)
        scores.masked_fill_(torch.arange(c)[None,None,:] > (c-q+local)[:,None,None], -float('inf'))
        ref = scores.softmax(-1) @ value[offset:offset+c,0].float()
        errors.append((actual[ix].float()-ref).abs().max().item())
        offset += n*block
    assert max(errors) < 0.005
    tiles = (max(contexts)+15)//16
    assert active.shape[1] == tiles
    print(json.dumps(dict(query=queries,context=contexts,packed_shape=list(active.shape),
                          upstream_full_width_bitwise_equal=True if compare_old else None,
                          max_abs_error_vs_cpu_fp32=max(errors))),flush=True)
