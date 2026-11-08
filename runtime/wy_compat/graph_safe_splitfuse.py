# SPDX-License-Identifier: Apache-2.0
"""Graph-safe causal mask for 310P speculative verification on legacy SplitFuse."""
import torch


def causal_mask(seq_lens, query_lens, width):
    """Keep context lengths on device; query lengths define the captured layout."""
    positions = torch.cat([
        seq_lens[index:index + 1] - length
        + torch.arange(length, dtype=torch.int32, device=seq_lens.device)
        for index, length in enumerate(query_lens)
    ])
    columns = torch.arange(width, dtype=torch.int32, device=seq_lens.device)
    return torch.zeros((positions.numel(), width), dtype=torch.float16,
                       device=seq_lens.device).masked_fill_(
        columns.unsqueeze(0) > positions.unsqueeze(1), float('-inf'))


def install():
    import torch_npu
    from vllm.forward_context import get_forward_context
    from vllm.config.compilation import CUDAGraphMode
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
    from vllm_ascend._310p.attention.metadata_builder import get_query_lens_cpu
    from vllm_ascend.attention.attention_v1 import AscendMetadata, AscendAttentionState
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_spec

    original_mask = AttentionMaskBuilder310.get_splitfuse_mask
    original_call = ACLGraphWrapper.__call__
    buffers = {}

    def layout(metadata):
        values = get_query_lens_cpu(metadata)
        if values is None:
            raise RuntimeError('SplitFuse graph requires prebuilt host query lengths')
        if isinstance(values, torch.Tensor):
            if values.device.type != 'cpu':
                raise RuntimeError('query lengths must already be host metadata')
            values = values.tolist()
        return tuple(int(value) for value in values)

    def prepare(wrapper, *args, **kwargs):
        context = get_forward_context()
        if wrapper.runtime_mode == CUDAGraphMode.FULL and not context.capturing:
            metadata = context.attn_metadata
            if isinstance(metadata, dict) and metadata:
                metadata = next((item for item in metadata.values() if isinstance(item, AscendMetadata)), None)
                if metadata is not None and metadata.attn_state in (AscendAttentionState.ChunkedPrefill, AscendAttentionState.SpecDecoding):
                    qlens = layout(metadata)
                    width = AttentionMaskBuilder310.max_seqlen
                    lengths = metadata.seq_lens.to(device='npu')
                    mask = causal_mask(lengths, qlens, width)
                    mask = torch_npu.npu_format_cast(nd_to_nz_spec(mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
                    key = (qlens, width, lengths.device.index)
                    if key in buffers:
                        buffers[key].copy_(mask)
                    else:
                        buffers[key] = mask
        return original_call(wrapper, *args, **kwargs)

    @classmethod
    def get_splitfuse_mask(cls, metadata, device):
        context = get_forward_context()
        if context.capturing:
            key = (layout(metadata), cls.max_seqlen, device.index)
            if key not in buffers:
                raise RuntimeError('SplitFuse mask must be prepared before graph capture')
            return buffers[key]
        return original_mask(metadata, device)

    ACLGraphWrapper.__call__ = prepare
    AttentionMaskBuilder310.get_splitfuse_mask = get_splitfuse_mask
    print('[splitfuse-fix] prepare stable mask buffer before FULL capture/replay', flush=True)
