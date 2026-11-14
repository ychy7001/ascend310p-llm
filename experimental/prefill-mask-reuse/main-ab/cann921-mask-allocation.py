# SPDX-License-Identifier: Apache-2.0
"""Compare the pinned upstream and proposed mask builders on legacy SplitFuse."""
import ast
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu


def load(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(selected) == len(names), (path, names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), namespace)


ns = dict(torch=torch, torch_npu=torch_npu, AscendMetadata=SimpleNamespace, ACL_FORMAT_FRACTAL_NZ=29,
          COMPRESSED_MASK_SEQ_LEN=2048, PAGED_ATTENTION_COMPRESSED_MASK_VALUE=-10000)
load('/workbench/upstream-utils.py', {'nd_to_nz_spec'}, ns)
for variant in ('upstream', 'candidate'):
    load(f'/workbench/{variant}.py', {'AttentionMaskBuilder310', 'is_compressed_mask_supported'}, ns)
    ns[variant] = ns['AttentionMaskBuilder310']
old, new = ns['upstream'], ns['candidate']
print(json.dumps({'compressed_symbols_present': ns['is_compressed_mask_supported'](), 'scope': 'legacy mask builder only', 'torch': torch.__version__, 'torch_npu': torch_npu.__version__}), flush=True)
torch.set_num_threads(1)
torch.manual_seed(20261114)
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = True
device = torch.device('npu:0')
limit = 12288
old.max_seqlen = new.max_seqlen = limit
q_lens, c_lens = [1, 2046, 1], [4097, 8192, 1025]
meta = SimpleNamespace(query_start_loc=torch.tensor([0,1,2047,2048],dtype=torch.int32,device=device),
                       seq_lens=torch.tensor(c_lens,dtype=torch.int32,device=device))
assert sum(q_lens) == 2048

def build(active):
    if active:
        return new.get_splitfuse_mask(meta, device, active_width=True)
    return old.get_splitfuse_mask(meta, device)

def measure(active):
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    mask = build(active)
    torch.npu.synchronize()
    return dict(ms=(time.perf_counter()-start)*1e3, peak_delta_bytes=torch.npu.max_memory_allocated()-before,
                live_delta_bytes=torch.npu.memory_allocated()-before, packed_shape=list(mask.shape),
                packed_bytes=mask.numel()*mask.element_size())

# Initialize dispatch for both shapes before timing first allocation at the target size.
small = SimpleNamespace(query_start_loc=torch.tensor([0,1],dtype=torch.int32,device=device),
                        seq_lens=torch.tensor([16],dtype=torch.int32,device=device))
old.max_seqlen = 16
old.get_splitfuse_mask(small,device)
new.get_splitfuse_mask(small,device,active_width=True)
torch.npu.synchronize()
old.chunked_prefill_attn_mask = None
old.max_seqlen = limit
first = {}
for active in (False,True):
    old.chunked_prefill_attn_mask = None
    first[str(active)] = measure(active)
steady = {'False':[], 'True':[]}
old_mask = build(False)
active_mask = build(True)
assert torch.equal(old_mask.cpu()[:,:512].contiguous(),active_mask.cpu().contiguous())
del old_mask, active_mask
for i in range(9):
    for active in ((False,True) if i%2 == 0 else (True,False)):
        steady[str(active)].append(measure(active))
print(json.dumps(dict(max_model_len=limit,query_lengths=q_lens,ctx_lengths=c_lens,first=first,steady=steady,
                     steady_median_ms={k:statistics.median(x['ms'] for x in v) for k,v in steady.items()},
                     baseline_persistent_square_bytes=old.chunked_prefill_attn_mask.numel()*old.chunked_prefill_attn_mask.element_size(),
                     candidate_persistent_square=new.chunked_prefill_attn_mask is not None), indent=2),flush=True)
