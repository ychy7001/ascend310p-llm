# SPDX-License-Identifier: Apache-2.0
"""310P ND C8 cache lifecycle component; not a registered vLLM backend.

Requires libqwen36_c8.so to be loaded by the caller. Quantization is performed
on the input device and the scale row is padded to 32 bytes for Ascend C DMA.
"""
import torch
from c8_backend_contract import C8PageSpec, validate_cache_tensors


class C8Cache310:
    def __init__(self, num_blocks, block_size, heads, dim, device='npu'):
        if not (num_blocks > 0 and block_size > 0 and 1 <= heads <= 16
                and dim > 0 and dim % 32 == 0 and heads * dim <= 32768):
            raise ValueError('unsupported cache dimensions')
        self.block_size, self.heads, self.dim = block_size, heads, dim
        self.capacity = num_blocks * block_size
        self.k = torch.zeros((self.capacity, heads, dim), dtype=torch.int8, device=device)
        self.v = torch.zeros_like(self.k)
        self.ks = torch.zeros((self.capacity, 16), dtype=torch.float16, device=device)
        self.vs = torch.zeros_like(self.ks)
        validate_cache_tensors(self.k, self.v, self.ks, self.vs, self.capacity,
                               C8PageSpec(block_size, heads, dim))

    def write(self, key, value, slots):
        if key.shape != value.shape or key.shape[1:] != (self.heads, self.dim):
            raise ValueError('expected matching [tokens, KV heads, head dim]')
        if key.shape[0] != slots.numel() or key.shape[0] > 4096:
            raise ValueError('slot count mismatch or write chunk exceeds 4096 tokens')
        for source, cache, scale_cache in ((key, self.k, self.ks), (value, self.v, self.vs)):
            # Quantize with the scale actually stored in the cache.  FP16's
            # smallest normal value also avoids device-dependent subnormal flush.
            scale32 = (source.float().abs().amax(-1) / 127).clamp_min(
                torch.finfo(torch.float16).tiny).half().float()
            quant = (source.float() / scale32.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
            scales = torch.zeros((source.shape[0], 16), dtype=torch.float16, device=source.device)
            scales[:, :self.heads] = scale32.half()
            torch.ops.npu.c8_page_scatter(quant.contiguous(), scales,
                                         slots.to(device=source.device, dtype=torch.int32).contiguous(),
                                         cache, scale_cache, source.shape[0], self.heads, self.dim)

    def read_page(self, physical_block, count=None):
        count = self.block_size if count is None else count
        start = physical_block * self.block_size
        if not (0 < count <= self.block_size and 0 <= start and start + count <= self.capacity):
            raise ValueError('page outside cache')
        return tuple(torch.ops.npu.c8_page_dequant(cache[start:start + count], scales[start:start + count],
                                                  count, self.heads, self.dim)
                     for cache, scales in ((self.k, self.ks), (self.v, self.vs)))

    def read_table_page(self, block_table, logical_page, count=None):
        """Resolve the physical page on device from a contiguous int32 table."""
        count = self.block_size if count is None else count
        return tuple(torch.ops.npu.c8_table_dequant(
            cache, scales, block_table, logical_page, self.block_size,
            count, self.heads, self.dim)
            for cache, scales in ((self.k, self.ks), (self.v, self.vs)))

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.ks, self.vs))

    def decode_causal_fused(self, query, block_table, sequence_length, cube=False, splits=1):
        """Opt-in fused vector prototype; QHD half query, device int32 table.

        This path is gated independently and is not a registered vLLM backend.
        """
        return torch.ops.npu.c8_attention(query, self.k, self.v, self.ks, self.vs,
                                         block_table, self.block_size, sequence_length, cube, splits)

    def decode_reference(self, query, block_table, sequence_length):
        """Page-local GQA online softmax; correctness path, not a fused kernel.

        query: [query_heads, dim], heads grouped contiguously per KV head.
        block_table may be host metadata or an int32 NPU tensor. Device tables
        are read directly by c8_table_dequant, without a device-to-host copy.
        """
        if query.ndim != 2 or query.shape[1] != self.dim or query.shape[0] % self.heads:
            raise ValueError('expected [query heads divisible by KV heads, head dim]')
        return self.decode_causal_reference(query.unsqueeze(0), block_table, sequence_length)[0]

    def decode_causal_reference(self, query, block_table, sequence_length):
        """Page-local causal verification for the final Q tokens, including MTP.

        query is [Q, query_heads, dim]; the cache already includes those Q tokens.
        This is a numerical reference, not the fused vLLM attention backend.
        """
        if (query.ndim != 3 or query.shape[-1] != self.dim
                or query.shape[1] % self.heads or query.shape[0] == 0):
            raise ValueError('expected [Q, query heads divisible by KV heads, head dim]')
        if not 0 < sequence_length <= len(block_table) * self.block_size:
            raise ValueError('sequence length exceeds block table')
        count_query, query_heads, _ = query.shape
        if count_query > sequence_length:
            raise ValueError('query count exceeds sequence length')
        groups = query_heads // self.heads
        q = query.reshape(count_query, self.heads, groups, self.dim).permute(1, 0, 2, 3)
        q = q.reshape(self.heads, count_query * groups, self.dim).float()
        visible = sequence_length - count_query + torch.arange(
            count_query, device=query.device) + 1
        visible = visible.repeat_interleave(groups)
        maximum = torch.full((self.heads, count_query * groups, 1), -float('inf'), device=query.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q)
        remaining = sequence_length
        device_table = isinstance(block_table, torch.Tensor) and block_table.device == self.k.device
        for logical_page in range(len(block_table)):
            if remaining == 0:
                break
            count = min(self.block_size, remaining)
            if device_table:
                k, v = self.read_table_page(block_table, logical_page, count)
            else:
                k, v = self.read_page(int(block_table[logical_page]), count)
            scores = torch.matmul(q, k.transpose(0, 1).float().transpose(1, 2)) * self.dim ** -0.5
            positions = sequence_length - remaining + torch.arange(count, device=query.device)
            scores = scores.masked_fill(positions[None, :] >= visible[:, None], -float('inf'))
            next_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
            alpha = (maximum - next_maximum).exp()
            probabilities = (scores - next_maximum).exp()
            numerator = numerator * alpha + torch.matmul(probabilities, v.transpose(0, 1).float())
            denominator = denominator * alpha + probabilities.sum(-1, keepdim=True)
            maximum = next_maximum
            remaining -= count
        return (numerator / denominator).reshape(self.heads, count_query, groups, self.dim).permute(
            1, 0, 2, 3).reshape_as(query)
