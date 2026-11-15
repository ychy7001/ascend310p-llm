# [Bug]: 310P compressed SplitFuse reads future V at context length 2049

### Your current environment

Atlas 300I DUO (Ascend 310P3), reproduced independently on both chips. CANN 9.2.1, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post7. The image has vllm-ascend revision `76ea9d24f155afe032ad9ac6c98717f1f7334f36`. The reproducer calls torch-npu directly and does not import vLLM, load weights, start a service, or use graph capture; these are direct operator measurements, not a serving test at `main`.

The dispatch and mask construction quoted below were read from vllm-ascend `main` at `4b3bd7ebe51094f3afa5288bb7a3316d12d3154e`.

### Describe the bug

With FP16, physical KV block 64, 8 query heads, 1 KV head, head dimension 256, and query length 384, the compressed operator violates causality for context length 2049. It is given the same fixed `[2048, 2048]` upper-triangular ND mask and `mask_type=5` that the serving path passes (`vllm_ascend/_310p/attention/attention_mask.py:24`, `vllm_ascend/_310p/attention/attention_v1.py:44`).

Run the [standalone reproducer](https://github.com/ychy7001/ascend310p-llm/blob/main/experimental/wy-compressed/upstream-submission/310p_splitfuse_causal_repro.py) in an initialized 310P torch-npu environment:

```bash
python 310p_splitfuse_causal_repro.py --device 0
```

Q and K are zero. For query row 123, change only the 16 V tokens at offsets +4 through +19 relative to that row's causal cutoff, from zero to 64. The target row must stay zero. A final-row positive control verifies that the new values reached the KV cache.

| Context | Legacy target row before → after | Compressed target row before → after | Compressed last-row control |
| --- | --- | --- | --- |
| 2048 | 0 → 0 | 0 → 0 | 0 → 0.5 |
| 2049 | 0 → 0 | 0 → 0.5673828125 | 0 → 0.5 |

Both chips give the same results, and the script returns 1 because the compressed invariant fails. A separate per-key probe confirms that at 2049 the target row still rejects offsets +1 through +3, but admits +4 through +19.

### Why this is not limited to an unusual shape

Two points, so the repro is not read as a mismatch with the default configuration.

- **Block size.** 64 is one of the two supported 310P block sizes, but the backend's own comment records 128 as the 310P default, chosen over 64 to avoid a paged decode regression (`vllm_ascend/_310p/attention/attention_v1.py:102-107`). A block-128, head-dim-128 variant still fails: `context=32774, query=384` admits future keys in 23/384 rows.
- **Dispatch.** The chunked-prefill path selects the compressed operator from a capability check alone. `is_compressed_mask_supported()` is two `hasattr` tests (`attention_mask.py:28`); `forward_chunked_prefill_310` branches on that flag with no context-length or history-alignment condition (`attention_v1.py:289`); the metadata builder installs the same fixed mask under the same condition (`metadata_builder.py:146`). Nothing on this path rejects a context above 2048 or a prefill history that is not page-aligned. We also did not find a configuration switch for this path in `vllm_ascend/ascend_config.py` or `vllm_ascend/_310p/worker_310p.py`; if one exists, please point us to it, since it would let affected deployments opt out meanwhile.

At this geometry (block 64, head dimension 256), a grid of 341 distinct shapes reproduced in 224. Every observed failure had `context > 2048` with `history = context - query` not a multiple of 64, but not every shape meeting both conditions failed, so neither condition is sufficient.

### Alignment is not a usable condition

Two separate observations, which are easy to conflate:

- **Total context alignment is neither necessary nor sufficient.** `context=32768, query=383` fails in 40/383 rows even though the total context is aligned; `context=32768, query=384` passes. The quantity that follows the measurements is the prefill history modulo the physical page, not the context length.
- **Even page-aligned history is not a guarantee.** With block 64 and head dimension 128, `context=32768, query=384` — history exactly page-aligned — still fails in 180/384 rows.

So the observed pattern should not be turned into a fallback predicate or an API precondition.

### Requested follow-up

Could the maintainers confirm the supported context/geometry contract for the ATB `PagedAttentionOperation` compressed mask path, and help route the operator issue to the ATB maintainers if appropriate? The direct torch-npu reproduction and the legacy control isolate the behavior below the vLLM scheduler. We do not know which internal tiling or mask indexing step fails.

A standalone reproducer with a legacy control is ready; I can open a Draft PR with it on request.

Related: #10181 added the compressed path. #11907 concerns stack/model compatibility when enabling compression; this report concerns causal output after the operator already runs.
