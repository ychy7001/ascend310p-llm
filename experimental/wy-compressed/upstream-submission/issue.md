# [Bug]: 310P compressed SplitFuse reads future V at context length 2049

### Your current environment

Atlas 300I DUO (Ascend 310P3), reproduced independently on both chips. CANN 9.2.1, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post7. The NPU run used a retained image with vllm-ascend revision `76ea9d24f155afe032ad9ac6c98717f1f7334f36`. It calls torch-npu directly, without importing vLLM, loading weights, starting a service, or using graph capture.

Separately, source review of vllm-ascend `main` at `4b3bd7ebe51094f3afa5288bb7a3316d12d3154e` confirms that the 310P chunked prefill path still selects `_npu_paged_attention_splitfuse_v2` when compressed-mask support is detected. We have not run a full serving test at that revision.

### Describe the bug

With FP16, physical KV block 64, 8 query heads, 1 KV head, head dimension 256, and query length 384, the compressed operator violates causality for context length 2049. It uses the same `[2048, 2048]` ND mask and `mask_type=5` as the serving path.

Run the [standalone reproducer](https://github.com/ychy7001/ascend310p-llm/blob/main/experimental/wy-compressed/upstream-submission/310p_splitfuse_causal_repro.py) in an initialized 310P torch-npu environment:

```bash
python 310p_splitfuse_causal_repro.py --device 0
```

Q and K are zero. For query row 123, change only the 16 V tokens at offsets +4 through +19 relative to that row's causal cutoff, from zero to 64. The target row must stay zero. A final-row positive control verifies that the new values reached the KV cache.

| Context | Legacy target row before → after | Compressed target row before → after | Compressed last-row control |
| --- | --- | --- | --- |
| 2048 | 0 → 0 | 0 → 0 | 0 → 0.5 |
| 2049 | 0 → 0 | 0 → 0.5673828125 | 0 → 0.5 |

Both chips give the same results, and the script returns 1 because the compressed invariant fails. A separate per-key probe confirms that at 2049 the target row still rejects offsets +1 through +3, but admits +4 through +19. In the model's block-64/head-dim-256 geometry a separate case fails with **aligned total context** (`32768`, query length `383`), while `32768/384` passes. The pattern should not be used as a general fallback predicate: different head/page geometry can also fail with aligned history.

### Requested follow-up

Could the maintainers confirm the supported context/geometry contract for the ATB `PagedAttentionOperation` compressed mask path, and help route the operator issue to the ATB maintainers if appropriate? The direct torch-npu reproduction and the legacy control isolate the behavior below the vLLM scheduler. We do not know which internal tiling or mask indexing step fails.

Related: #10181 added the compressed path. #11907 concerns stack/model compatibility when enabling compression; this report concerns causal output after the operator already runs.
