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
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor
import vllm.v1.core.kv_cache_utils as cache_utils
# Initialize the ops package before DeviceOperator imports its submodules.
# Importing the runner first creates a DeviceOperator/fused_moe cycle.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackendImpl310
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from c8_cache_310 import C8Cache310

torch.ops.load_library(os.environ['QWEN36_C8_LIBRARY'])
_SCRATCH = {}
_NZ_READY = set()

# Explicit 310P experiment contract: Qwen full-attention geometry and the
# device-side page size are intentionally named instead of repeated literals.
C8_BLOCK_SIZE = 64
C8_KV_HEADS = 2
C8_HEAD_SIZE = 256


def _flag(name, default='0'):
    return os.environ.get(name, default) == '1'


def _graph_candidate_enabled():
    return _flag('QWEN36_C8_ALLOW_GRAPH')


def _mtp2_enabled():
    return _flag('QWEN36_C8_ALLOW_MTP2')


def _persistent_nz_enabled():
    return _flag('QWEN36_C8_PERSISTENT_NZ')


def _c8_device_context_length(seq_lens):
    """Return the scalar shape required by the 310P C8 conversion op.

    The product profile is TP1 with one active request.  During FULL graph
    capture vLLM can retain a padded/non-contiguous sequence-length vector,
    while the custom op accepts exactly one contiguous INT32 value.
    """
    if seq_lens.numel() < 1:
        raise RuntimeError('C8 device context length requires one active sequence')
    if (seq_lens.numel() == 1 and seq_lens.dtype == torch.int32
            and seq_lens.is_contiguous()):
        return seq_lens
    return seq_lens.reshape(-1)[:1].to(dtype=torch.int32).contiguous()

# The pinned image wraps FULL graphs even with compilation mode NONE, but
# initializes GraphParams only for VLLM_COMPILE. Keep both conditions aligned
# for the explicit direct-capture probe; never bypass parameter updates.
if _graph_candidate_enabled():
    import direct_graph_310  # noqa: F401


@dataclass(frozen=True, kw_only=True)
class C8FullAttentionSpec(FullAttentionSpec):
    @property
    def real_page_size_bytes(self):
        # Two signed-byte data tensors and two padded FP16 scale tensors.
        return self.block_size * (2 * self.num_kv_heads * self.head_size + C8_BLOCK_SIZE)


_original_spec = Attention.get_kv_cache_spec
def get_spec(self, config):
    spec = _original_spec(self, config)
    if spec is None or not isinstance(self.impl, AscendAttentionBackendImpl310):
        return spec
    graph_candidate = _graph_candidate_enabled()
    if (_persistent_nz_enabled() and not config.model_config.enforce_eager):
        raise RuntimeError('persistent NZ prototype is eager-only; disable Graph')
    if (config.parallel_config.tensor_parallel_size != 1
            or (not config.model_config.enforce_eager and not graph_candidate)
            or config.kv_transfer_config):
        raise RuntimeError('C8 experiment requires TP1/eager/no-KV-transfer; set QWEN36_C8_ALLOW_GRAPH=1 only for graph experiments')
    speculative = config.speculative_config
    if speculative and (speculative.method != 'mtp'
            or speculative.num_speculative_tokens not in (1, 2)
            or (speculative.num_speculative_tokens == 2
                and not _mtp2_enabled())):
        raise RuntimeError('C8 supports MTP1 by default; set QWEN36_C8_ALLOW_MTP2=1 for the experimental MTP2 path')
    if not isinstance(spec, FullAttentionSpec) or spec.head_size != C8_HEAD_SIZE or spec.num_kv_heads != C8_KV_HEADS:
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
    scratch = math.ceil(config.model_config.max_model_len / C8_BLOCK_SIZE) * C8_BLOCK_SIZE * 2 * C8_HEAD_SIZE * 4
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
    # All cache groups draw block IDs from the same scheduler pool. Full
    # attention pages alone omit Mamba running/speculative/rotation state and
    # the pool's reserved null block. Use each spec's upstream worst-case
    # contract, including every group even when their physical tensors share.
    scheduler_required_blocks = 1 + sum(
        math.ceil(group.kv_cache_spec.max_memory_usage_bytes(config)
                  / group.kv_cache_spec.page_size_bytes)
        for group in cfg.kv_cache_groups
    )
    # Each C8 physical pool must contain every logical page of a request.
    # Use the scheduler page size (1152 for this hybrid model), not the
    # 64-token kernel subblock size used by the expanded device block table.
    needed = max(required_blocks, scheduler_required_blocks)
    print(f'C8_BUDGET mode={pool_mode} groups={len(cfg.kv_cache_groups)} layers={len(layer_specs)} '
          f'original_blocks={cfg.num_blocks} independent_blocks={blocks} minimum_blocks={needed} '
          f'charged_bytes_per_block={per_block} required_context_blocks={required_blocks} '
          f'scheduler_required_blocks={scheduler_required_blocks} '
          f'original_budget_bytes={old_budget}',flush=True)
    if blocks < needed:
        raise RuntimeError('C8 allocation cannot fit full context plus hybrid/speculative scheduler state in budget')
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


def scratch_buffers(device, context, heads, dim, layer_key=None):
    # A persistent workspace is one per attention layer. The legacy path
    # deliberately shares one workspace because it rewrites every page before
    # each layer; sharing is invalid for incremental page updates.
    key = (str(device), context, heads, dim, layer_key)
    if key not in _SCRATCH:
        if layer_key is not None:
            max_gib = float(os.environ.get('QWEN36_C8_PERSISTENT_NZ_MAX_GIB', '4'))
            expected = math.ceil(context / C8_BLOCK_SIZE) * (heads * dim // 16) * 64 * 16 * 2 * 2
            expected *= 41
            if expected > max_gib * (1 << 30):
                raise RuntimeError(f'persistent NZ estimate {expected / (1 << 30):.2f} GiB exceeds '
                                   f'QWEN36_C8_PERSISTENT_NZ_MAX_GIB={max_gib:g}')
        blocks = math.ceil(context / C8_BLOCK_SIZE)
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
    cache.block_size = C8_BLOCK_SIZE
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
    layer_key = layer.layer_name
    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
        _NZ_READY.discard(layer_key)
        return self.forward_prefill_310(query,key,value,attn_metadata,output)
    length = attn_metadata.seq_lens_list[0]
    context = self.vllm_config.model_config.max_model_len
    # Python sequence lengths do not change during graph replay. Convert the
    # fixed max-context table only for a forward dispatched to a FULL graph;
    # native PA masks by device context_lens. Chunked prefill remains eager
    # even in a graph-enabled service and must use its actual history length.
    graph_candidate = (_graph_candidate_enabled()
                       and not self.vllm_config.model_config.enforce_eager
                       and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL)
    pages = math.ceil((context if graph_candidate else length) / C8_BLOCK_SIZE)
    if attn_metadata.block_tables.shape[1] < pages:
        raise RuntimeError('C8 graph requires a max-context device block table')
    selected = attn_metadata.block_tables[0, :pages]
    if (os.environ.get('QWEN36_C8_REUSE_INT32_METADATA', '0') != '1'
            or selected.device.type != 'npu'
            or selected.dtype != torch.int32
            or not selected.is_contiguous()):
        selected = selected.to(device=query.device, dtype=torch.int32).contiguous()
    persistent_nz = _flag('QWEN36_C8_PERSISTENT_NZ')
    layer_workspace_key = layer_key if persistent_nz else None
    ko,vo,identity = scratch_buffers(query.device,context,cache.heads,cache.dim,layer_workspace_key)
    compact_state = attn_metadata.attn_state == AscendAttentionState.DecodeOnly
    direct_page = (persistent_nz and not graph_candidate and compact_state
                   and layer_key in _NZ_READY and count <= 1)
    # Any chunked prefill or multi-token verify changes more than one page and
    # invalidates the incremental state; rebuild the full workspace.
    if not compact_state or not direct_page:
        _NZ_READY.discard(layer_key)
    target_ko, target_vo = (ko, vo) if direct_page else (ko[:pages], vo[:pages])
    if graph_candidate and os.environ.get('QWEN36_C8_DEVICE_LENGTH', '0') == '1':
        # Same replay-updated tensor that native PA consumes. Keep output and
        # page table extent fixed; only device-side conversion work varies.
        device_context_length = _c8_device_context_length(attn_metadata.seq_lens)
        if direct_page:
            torch.ops.npu.c8_cache_to_nz_range(cache.k,cache.v,cache.ks,cache.vs,target_ko,target_vo,
                                               C8_BLOCK_SIZE,selected[pages-1:pages].contiguous(),pages-1)
        else:
            torch.ops.npu.c8_cache_to_nz(cache.k,cache.v,cache.ks,cache.vs,target_ko,target_vo,C8_BLOCK_SIZE,selected,
                                         device_context_length)
    else:
        if direct_page:
            torch.ops.npu.c8_cache_to_nz_range(cache.k,cache.v,cache.ks,cache.vs,target_ko,target_vo,
                                               C8_BLOCK_SIZE,selected[pages-1:pages].contiguous(),pages-1)
        else:
            torch.ops.npu.c8_cache_to_nz(cache.k,cache.v,cache.ks,cache.vs,target_ko,target_vo,C8_BLOCK_SIZE,selected)
    if persistent_nz:
        _NZ_READY.add(layer_key)
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
