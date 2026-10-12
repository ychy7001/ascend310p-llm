# SPDX-License-Identifier: Apache-2.0
"""TP1/eager C8 experiment with explicit independent layer buffers.

Uses a real C8 spec and charged scale storage. The first allocator retains
hybrid page padding and replaces shared pools by independent layer buffers,
reducing scheduler blocks to stay inside its original memory budget. This is
an explicit correctness candidate. Physical C8 buffers omit unused tail
padding while the scheduler conservatively retains its charged page budget.
This does not increase advertised context capacity.
"""
from dataclasses import dataclass, replace
from copy import copy
import math
import os
import torch
import torch_npu
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor
import vllm.v1.core.kv_cache_utils as cache_utils
# Initialize the ops package before DeviceOperator imports its submodules.
# Importing the runner first creates a DeviceOperator/fused_moe cycle.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackendImpl310
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from c8_cache_310 import C8Cache310

torch.ops.load_library(os.environ['QWEN36_C8_LIBRARY'])
_SCRATCH = {}


@dataclass(frozen=True, kw_only=True)
class C8FullAttentionSpec(FullAttentionSpec):
    @property
    def real_page_size_bytes(self):
        # Two signed-byte data tensors and two padded FP16 scale tensors.
        return self.block_size * (2 * self.num_kv_heads * self.head_size + 64)


_original_spec = Attention.get_kv_cache_spec
def get_spec(self, config):
    spec = _original_spec(self, config)
    if spec is None or not isinstance(self.impl, AscendAttentionBackendImpl310):
        return spec
    if (config.parallel_config.tensor_parallel_size != 1
            or not config.model_config.enforce_eager or config.kv_transfer_config):
        raise RuntimeError('C8 experiment requires TP1/eager/no-KV-transfer')
    speculative = config.speculative_config
    if speculative and (speculative.method != 'mtp' or speculative.num_speculative_tokens != 1):
        raise RuntimeError('C8 speculative experiment supports only MTP with one draft token')
    if not isinstance(spec, FullAttentionSpec) or spec.head_size != 256 or spec.num_kv_heads != 2:
        raise RuntimeError('C8 bring-up supports Qwen full attention 2KV/dim256 only')
    return C8FullAttentionSpec(block_size=spec.block_size, num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size, head_size_v=spec.head_size_v, dtype=torch.int8,
        page_size_padded=spec.page_size_bytes)
Attention.get_kv_cache_spec = get_spec


_original_configs = cache_utils.get_kv_cache_configs
def get_configs(config, specs, memory):
    configs = _original_configs(config, specs, memory)
    if not any(isinstance(s, C8FullAttentionSpec) for worker in specs for s in worker.values()):
        return configs
    if len(configs) != 1:
        raise RuntimeError('C8 bring-up allocator is TP1 only')
    cfg = configs[0]
    layer_specs = {name: group.kv_cache_spec for group in cfg.kv_cache_groups
                   for name in group.layer_names}
    # Reserve one shared, max-context FP16 NZ workspace, not one per layer.
    scratch = math.ceil(config.model_config.max_model_len / 64) * 64 * 2 * 256 * 4
    old_budget = sum(t.size for t in cfg.kv_cache_tensors)
    pool_mode = os.environ.get('QWEN36_C8_POOL_MODE','independent')
    if pool_mode not in ('independent','split_by_type'):
        raise RuntimeError('C8 pool mode must be independent or split_by_type')
    pools=[]
    if pool_mode == 'split_by_type':
        # Upstream assigns disjoint block IDs to layers in different groups
        # sharing one pool. Preserve that sharing within each storage type;
        # split C8 payloads from the original linear-state padded buffers.
        for tensor in cfg.kv_cache_tensors:
            for c8 in (False,True):
                names=[name for name in tensor.shared_by
                       if isinstance(layer_specs[name],C8FullAttentionSpec)==c8]
                if not names: continue
                first=layer_specs[names[0]]
                if any(layer_specs[name] != first for name in names):
                    raise RuntimeError('C8 split pool requires identical specs within each type')
                size=first.real_page_size_bytes if c8 else first.page_size_bytes
                pools.append((names,size))
    else:
        pools=[([name],spec.page_size_bytes) for name,spec in layer_specs.items()]
    per_block = sum(size for _,size in pools)
    blocks = min(cfg.num_blocks, (old_budget - scratch) // per_block)
    full = next(s for s in layer_specs.values() if isinstance(s, C8FullAttentionSpec))
    required_blocks = math.ceil(config.model_config.max_model_len / full.block_size)
    # Each C8 physical pool must contain every logical page of a request.
    # The old check used the number of hybrid groups, which could let a 32K
    # config start with only ~5K physical slots and fail later on block tables.
    needed = required_blocks
    print(f'C8_BUDGET mode={pool_mode} groups={len(cfg.kv_cache_groups)} layers={len(layer_specs)} '
          f'original_blocks={cfg.num_blocks} independent_blocks={blocks} minimum_blocks={needed} '
          f'charged_bytes_per_block={per_block} required_context_blocks={required_blocks} '
          f'original_budget_bytes={old_budget}',flush=True)
    if blocks < needed:
        raise RuntimeError('Independent C8 layer allocation cannot fit requested context in budget')
    cfg.num_blocks = blocks
    cfg.kv_cache_tensors = [KVCacheTensor(size=blocks*size,shared_by=names) for names,size in pools]
    print(f'C8_ALLOCATOR mode={pool_mode} layers={len(layer_specs)} pools={len(pools)} blocks={blocks} '
          f'allocated_bytes={sum(t.size for t in cfg.kv_cache_tensors)} '
          f'reserved_scratch_bytes={scratch} original_budget_bytes={old_budget}', flush=True)
    return configs
cache_utils.get_kv_cache_configs = get_configs
import vllm.v1.engine.core as engine_core
engine_core.get_kv_cache_configs = get_configs


_original_allocate = NPUModelRunner310._allocate_kv_cache_tensors
def allocate(self, config):
    specs = {name:g.kv_cache_spec for g in config.kv_cache_groups for name in g.layer_names}
    filtered = replace(config,
        kv_cache_groups=[g for g in config.kv_cache_groups if not isinstance(g.kv_cache_spec,C8FullAttentionSpec)],
        kv_cache_tensors=[t for t in config.kv_cache_tensors
                         if not isinstance(specs[t.shared_by[0]],C8FullAttentionSpec)])
    result = _original_allocate(self, filtered)
    buffers = {name:t for t in config.kv_cache_tensors for name in t.shared_by}
    allocated={}
    for name,spec in specs.items():
        if not isinstance(spec, C8FullAttentionSpec):
            continue
        allocation = buffers[name]
        pool_key=tuple(allocation.shared_by)
        if pool_key in allocated:
            result[name]=allocated[pool_key]
            continue
        if any(not isinstance(specs[other],C8FullAttentionSpec) for other in allocation.shared_by):
            raise RuntimeError('C8 physical pool cannot alias linear state')
        blocks = config.num_blocks
        capacity = blocks * spec.block_size
        data_bytes = capacity * spec.num_kv_heads * spec.head_size
        scale_bytes = capacity * 16 * 2
        used = 2*data_bytes + 2*scale_bytes
        if used > allocation.size:
            raise RuntimeError('C8 data and scales exceed charged allocation')
        # Views consume a contiguous payload, never the hybrid padded tail.
        # Keep the scheduler charge unchanged; this saves physical storage
        # without advertising extra blocks that its grouping cannot support.
        tensor = torch.zeros(used,dtype=torch.int8,device=self.device)
        k = tensor[:data_bytes].view(capacity, spec.num_kv_heads, spec.head_size)
        v = tensor[data_bytes:2*data_bytes].view_as(k)
        ks = tensor[2*data_bytes:2*data_bytes+scale_bytes].view(torch.float16).view(capacity,16)
        vs = tensor[2*data_bytes+scale_bytes:used].view(torch.float16).view_as(ks)
        result[name] = (k,v,ks,vs)
        allocated[pool_key]=result[name]
        print(f'C8_STORAGE layer={name} slots={capacity} data_scale_bytes={used} '
              f'allocated_bytes={tensor.numel()} charged_bytes={allocation.size} '
              f'avoided_padding_bytes={allocation.size-used}',flush=True)
    return result
NPUModelRunner310._allocate_kv_cache_tensors = allocate


def scratch_buffers(device, context, heads, dim):
    key = (str(device), context, heads, dim)
    if key not in _SCRATCH:
        blocks = math.ceil(context/64)
        shape = (blocks, heads*dim//16,64,16)
        k = torch_npu.empty_with_format(size=shape,dtype=torch.float16,device=device,acl_format=29)
        v = torch_npu.empty_with_format(size=shape,dtype=torch.float16,device=device,acl_format=29)
        table = torch.arange(blocks,dtype=torch.int32,device=device)
        _SCRATCH[key] = k,v,table
        print(f'C8_SCRATCH shared_bytes={4*k.numel()} context={context}',flush=True)
    return _SCRATCH[key]


def forward(self, layer, query, key, value, kv_cache, attn_metadata,
            output=None, output_scale=None, output_block_scale=None):
    if output is None or output_scale is not None or output_block_scale is not None:
        raise RuntimeError('C8 requires a plain preallocated output')
    if attn_metadata is None:
        return output.zero_()
    if len(kv_cache) != 4 or len(attn_metadata.seq_lens_list) != 1:
        raise RuntimeError('C8 requires real four-tensor storage and one request')
    count = attn_metadata.num_actual_tokens
    cache = C8Cache310.__new__(C8Cache310)
    cache.k,cache.v,cache.ks,cache.vs = kv_cache
    cache.heads,cache.dim = self.num_kv_heads,self.head_size
    cache.capacity = cache.k.shape[0]
    cache.block_size = 64
    if key is not None and value is not None:
        k = key[:count].reshape(count,cache.heads,cache.dim)
        v = value[:count].reshape_as(k)
        for start in range(0,count,4096):
            cache.write(k[start:start+4096],v[start:start+4096],
                        attn_metadata.slot_mapping[start:start+4096])
    if not getattr(self,'_c8_logged',False):
        print(f'C8_FORWARD layer={layer.layer_name} k_dtype={cache.k.dtype} '
              f'scale_dtype={cache.ks.dtype} state={attn_metadata.attn_state}',flush=True)
        self._c8_logged=True
    # First prefill uses current float inputs; all historical reads use C8.
    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
        return self.forward_prefill_310(query,key,value,attn_metadata,output)
    length = attn_metadata.seq_lens_list[0]
    pages = math.ceil(length/64)
    selected = attn_metadata.block_tables[0,:pages].to(torch.int32).contiguous()
    context = self.vllm_config.model_config.max_model_len
    ko,vo,identity = scratch_buffers(query.device,context,cache.heads,cache.dim)
    ko,vo = ko[:pages],vo[:pages]
    torch.ops.npu.c8_cache_to_nz(cache.k,cache.v,cache.ks,cache.vs,ko,vo,64,selected)
    # The 310P builder attaches query_lens_cpu dynamically; dataclasses.replace
    # discards it and forces an eager device->host copy in every SplitFuse layer.
    # Preserve the builder's full metadata, changing only the compact page table.
    compact = copy(attn_metadata)
    compact.block_tables = identity[:pages].reshape(1,-1)
    self.key_cache,self.value_cache = ko,vo
    q = query[:count].reshape(count,self.num_heads,self.head_size)
    if compact.attn_state == AscendAttentionState.DecodeOnly:
        return self.forward_paged_attention(q,compact,output)
    return self.forward_chunked_prefill_310(q,compact,output)
AscendAttentionBackendImpl310.forward = forward
