# Draft: Dense TP2 W8A8_DYNAMIC startup fails in QuantBatchMatmulV3

Status: unposted compatibility investigation; intermittent standalone failure reproduced with actual captured inputs; no confirmed source fix.

The public Eco-Tech Qwen3.8-27B-w8a8-310p checkpoint passes per-file pinned revision/SHA256 verification and safetensors index/header checks. The config resolves to Qwen3_5ForConditionalGeneration. On our Ascend 310P development host, both TP ranks load weights successfully, reporting 19.2416 GB each, but startup profiling fails before the HTTP server becomes healthy.

Environment is the same production image used for the 35B-A3B comparison: vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, vLLM-Ascend `88eac2713b14beedc7a48803274f8d6cee3e9f30`, torch 2.10.0+cpu, torch-npu 2.10.0.post2, CANN 9.1.0-beta.1, host driver 26.1.1. Tensor parallel size 2, FP16, ModelSlim W8A8_DYNAMIC, max context 262144, max batched tokens 4096, max sequences 4, prefix cache on, MTP off.

The initial asynchronous error reports 507015 and kernel `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21`. A separate synchronous eager run, with ACL Graph disabled, reproduces the device error and locates the call in MLP down_proj through `AscendW8A8DynamicLinearMethod310.apply`. An earlier attempted synchronous run was rejected at config validation because this runtime disallows ASCEND_LAUNCH_BLOCKING=1 with ACL Graph; that is a configuration failure, not an operator execution.

A diagnostic worker captured both ranks' first down_proj inputs before the call:

| Argument | Shape | Stride | Dtype | NPU format |
| --- | --- | --- | --- | --- |
| quantized activation | [4096, 8704] | [8704, 1] | int8 | ND (2) |
| weight | [8704, 5120] | [1, 8704] | int8 | FRACTAL_NZ (29) |
| weight scale | [5120] | [1] | float32 | ND (2) |
| per-token scale | [4096] | [1] | float32 | ND (2) |

The captured activation and per-token scale are entirely zero and finite. Weight values are in [-128, 127], scales are finite and positive. Capturing adds host synchronization; the model still fails startup. CPU snapshots are retained on the experimental host; they contain public model weights and are not copied into the repository.

In the initial limited screen, restoring these captured tensors in a new process and reconstructing the recorded layout passes all six tested variants on device0 using the rank0 capture: full 4096 rows in captured/ND/NZ activation layouts, and captured layout with 1/4/64 rows. CPU sampled integer dot products followed by explicit scales agree exactly (all zero). Synthetic random integer inputs also pass the original production weight layout at M=1/4096 for both MLP projection shapes. An alternative transpose-then-NZ layout fails numerical checks, and a plain ND weight layout is rejected by the API. Neither is a valid workaround.

Thus neither “zero per-token scale necessarily crashes the operator” nor “the logical shape alone is unsupported” is supported by the current evidence. Restoring CPU snapshots changes allocation, tensor descriptors and call history, which remain unresolved factors. The asynchronous Python stack must not be used to blame an unrelated later Mul/index operation. There is no OOM evidence in these failures.

A separate original-Graph startup screen with ACLNN_CACHE_LIMIT=0 also fails with 507015 in the same QuantBatchMatmulV3 kernel. It never reaches HTTP requests and is not an accepted baseline or performance result. Full model call order and runtime compatibility need further investigation before proposing a source patch.

Scripts: `quant_trace_worker.py`, `replay_dense_quant_capture.py`, `probe_dense_quant_matmul.py`, and their supervisors under `benchmarks/ascend310p/`. Results: `results/qwen38-dense-baseline-20261103/`, `results/qwen38-dense-baseline-sync-eager-diagnostic-20261103/`, `results/qwen38-dense-baseline-quant-trace-20261103/`, `results/dense-quant-captured-replay-v2-20261103/`.

Further isolation on November 4, 2026:

- Both actual TP captures were crossed with both physical devices in separate processes (rank0/device0, rank0/device1, rank1/device0, rank1/device1), 4096 rows each. All four passed the sampled CPU reference (96 elements, max error 0) and finite-output check. The captured activation and per-token scale are zero, so this is a startup-failure reproduction screen rather than a broad numerical quality test.
- Inside the complete model, rebuilding each down_proj weight via physical [N,K] ND -> fresh allocation -> NZ -> transpose preserved every logical int8 value exactly. Both ranks retained shape [8704,5120], stride [1,8704], format29, offset0, storage44564480 bytes. Rank0 completed the first call, but rank1 still failed at that call with the same 507015/kernel hash. Weight-storage rebuilding alone is not a workaround.
- A separately audited post4 image is being screened as a complete runtime compatibility comparison; it changes vLLM-Ascend and CANN as well, so it cannot identify torch-npu alone as a cause. The first attempt inherited the image entrypoint and failed reading /model before weight loading; it is retained as a launch failure, with an explicit-entrypoint retry tracked separately.

Additional evidence: `results/qwen38-dense-quant-repack-20261104/`, `results/dense-quant-rank-device-grid-20261104/`, `results/dense-runtime-audit-20261104/`.

Repeated standalone execution now reproduces the original failure. With `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, the first four-case cross had three passes and one 507015 failure (rank1 capture/device0), with the original QuantBatchMatmulV3 kernel/hash. An alternating five-repeat allocator control on rank1/device0 produced False: 4 PASS/1 FAIL507015; True: 3 PASS/1 FAIL507015, with the final True process stalled during device initialization in driver HDC receive and stopped after128.88s, before producing a tensor preparation/result marker. That last attempt is not counted as an operator failure. Both explicit allocator settings can fail, so disabling expandable segments is not a workaround. Earlier passing replays were insufficient to rule out an intermittent standalone issue.

A separate post4 runtime image (`b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d`) uses the same vLLM commit but Ascend `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`, torch-npu2.10.0.post4 and CANN install metadata9.1.0/V100R001C11SPC001B243. Its dynamic W8A8 Python source is byte-identical to the production version. After bypassing image-specific C8 bootstrap hooks, a synchronous eager attempt completed model `profile_run` without507015, then failed KV admission: 8.09GiB needed for262144 context, only7.66GiB available after the built-in workspace reserve at utilization0.90. A single successful profile does not prove the intermittent fault fixed, and this image-level screen cannot isolate a particular runtime component. No inference quality or performance result was obtained in that attempt.

Further raw evidence: `results/dense-quant-expandable-grid-20261104/`, `results/dense-quant-allocator-paired-20261104/`, `results/qwen38-dense-post4-runtime-v3-20261104/`. Original failed cleanup status is retained where restoring the temporary service hit systemd start limits; primary startup failures remain in server logs and separate adjudication.

The post4 utilization0.94 capacity screen then initialized262144 context successfully (reported KV274795tokens, rank0/1 budget8.50/8.63GiB) and served HTTP requests. Its strict short-question gate was7/8: sorting expected`2,7,9`, actual`279`; the A/B/A canary did not leak and before/after answers matched. The raw failed gate remains intact, no performance or broader accuracy acceptance is claimed. This is admission plus short requests, not an actual256K prompt test.

Separately, generated random int8 weights with zero activations/per-token scales reproduce507015 in the production image as well. A frozen probe using`--synthetic` reads no checkpoint or capture file; this can be shared without model/business data. Final repeated counts are tracked separately in the synthetic reproduction bundle.
