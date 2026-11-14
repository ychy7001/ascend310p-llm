# [Performance][310P] Bound legacy eager SplitFuse masks to the active batch

> 状态（2026-11-14）：legacy 分配补丁归档，暂停投稿。内部默认继续采用已验收的 WY 镜像；当前优化主线是该镜像的 eager forward mask 复用。本文保留为调查记录，不是待发布计划。

### What this PR does / why we need it?

Legacy 310P SplitFuse builds a square FP16 mask from the configured maximum sequence length, then selects rows for the current query batch. For eager chunked prefill and prefix hits, build the same causal values directly in a rectangle bounded by the active query count and largest context length. The legacy speculative verification and capture paths keep the existing width. Compressed SplitFuse and initial FlashAttention use their existing implementations.

For `query lengths = [17, 15, 1]`, `context lengths = [129, 65, 33]` and `max_model_len = 4096`, the result pads to 144 columns instead of 4096, without building the full square. This is scoped to legacy SplitFuse and does not, alone, add support for any new maximum model length.

### Does this PR introduce any user-facing change?

No API or configuration changes. Reduces mask allocation on legacy eager chunked prefill. Intended Attention outputs are unchanged.

### How was this patch tested?

- Three focused CPU tests passed with the pinned upstream layout functions and a mocked NPU format cast, covering mixed causal boundaries, NZ tile rounding, and preservation of the fixed-width builder. The full upstream test environment was not used for this run.
- Atlas 300I DUO operator test: single-request, mixed and reordered layouts produced bitwise-identical complete FP16 output versus the upstream builder (reference width 512). The active-width-only stress case reached 32,774 context tokens with CPU FP32 sample error `1.54e-4`. No full-width reference was allocated for that stress case. The same results were reproduced in the official `v0.30.0-310p` CANN 9.2.1 image, explicitly invoking the legacy kernel with the pinned upstream mask/layout functions.
- Isolated mask allocation at a configured width of 12,288 and an active width of 8,192 removed the 288 MiB persistent square per worker and reduced the packed mask from 48 to 32 MiB for 2,048 query tokens. These are mask allocations, not model or service memory savings.
- The pinned upstream Dockerfile runtime used compressed SplitFuse by default. With legacy SplitFuse forced in both arms, the first sampled chunk used 12,288 columns before and 3,840 columns after the patch on both ranks. Sampled device-memory peaks fell by 395/396 MB. Cold TTFT changed by less than 1% (three samples per arm and length), so no latency improvement is claimed. Prefix hits and 8K retrieval matched. The two concurrent text differences reproduced between unpatched runs; the repeat matched all four candidate responses.
- Forced-legacy MTP with full decode graph capture failed in the unpatched baseline: `get_splitfuse_mask` synchronously copies `query_start_loc` to the CPU during capture (`aclrtMemcpy` error 107030). No candidate graph result is claimed. Eager MTP2 admission passed in both arms (one 3,072-input/64-output request each). Both workers retained the 12,288-column SpecDecoding mask. Speculative acceptance counts differed, so this is an admission check, not a determinism or performance claim.

Historical context (separate environment): our earlier Qwen3.6 W8A8 service image locally integrated #17640 native WY and other runtime changes. Its mask reuse figures are not part of the main A/B benchmark for this allocation patch, and this PR does not include the WY change.
