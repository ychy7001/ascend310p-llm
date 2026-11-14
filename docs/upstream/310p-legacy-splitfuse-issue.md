# [310P] Legacy eager SplitFuse allocates a full-context causal mask for each worker

> 状态（2026-11-14）：legacy 分配补丁归档，暂停投稿。内部默认继续采用已验收的 WY 镜像；当前优化主线是该镜像的 eager forward mask 复用。本文保留为调查记录，不是待发布计划。

On the legacy 310P SplitFuse path, `AttentionMaskBuilder310.get_splitfuse_mask` lazily builds a persistent square FP16 causal mask at the configured maximum model length. It then selects rows for the current query chunk. The packed result also retains the configured width, even when the current batch only attends to a much shorter context. This is independent of model architecture.

I traced this on upstream `vllm-project/vllm-ascend` at `76ea9d24f155afe032ad9ac6c98717f1f7334f36`. The case requires the legacy operator (the compressed SplitFuse/FlashAttention capability check is false). On this path, at a configured maximum length of 131,072, the FP16 square alone would require 32 GiB per worker, before intermediate allocations. This is a size calculation, not a claim that a full upstream server reached this point at that length. The separate initial FlashAttention mask can also impose a long-context limit.

Minimal reproduction: on a 310P runner with the legacy operator, call the existing SplitFuse builder using a legal smaller `max_seqlen` such as 4096, `query_start_loc=[0, 17, 32, 33]` and `seq_lens=[129, 65, 33]`. The original returns a packed NZ mask 4096 columns wide. The active rows only need 129 columns (144 after NZ padding). This exercises the problem without risking a device OOM. Inspect the operator dispatch under `vllm_ascend/_310p/attention/attention_v1.py`; the compressed path bypasses this builder.

Proposed first patch: construct `[sum(query lengths), max(context lengths)]` directly for legacy eager chunked prefill/cache-hit. Keep compressed, speculative verification, and capture width unchanged. A separate optimization can share the resulting mask across layers in one forward; that follow-up should avoid host copies on every hit, use device-corrected lengths with async speculative decoding, and have its own A/B benchmark.

On Atlas 300I DUO, I ran the pinned upstream and proposed builder bodies through the same real legacy NPU operator. Single-request, mixed and reordered batches produced bitwise-equal FP16 attention outputs with fixed width 512 as reference. A separate three-request case reached 32,774 context tokens with the active-width mask and was checked against sampled CPU FP32 attention; the full-width reference was deliberately not allocated at that size. The result was reproduced with the official `v0.30.0-310p` CANN 9.2.1 image (`torch-npu 2.10.0.post7`), using the pinned upstream mask/layout functions and explicitly calling legacy SplitFuse. This was an operator test, not a serving run of upstream `main`.

A separate isolated allocation test at configured width 12,288, with 2,048 query tokens and active width 8,192, removed the 288 MiB persistent FP16 square per worker and reduced the packed mask from 48 to 32 MiB.

The pinned upstream 310P Dockerfile build (vLLM v0.33.0 and CANN 9.2.1) selected compressed SplitFuse in both tensor-parallel workers for the Qwen3.6-35B-A3B W8A8 chunked-prefill run. The proposed change only affects legacy SplitFuse. Are deployments that require the legacy operator still in scope for improvements? Any forced-legacy benchmark will be labeled separately from the default dispatch.
