# C8 Fused Attention Microbenchmark

Date: 2026-10-15, Atlas 300I / Ascend 310P, CANN 9.1, `ascend310p1`.

The experiment used the same C8 shared library as the TP1 product service and
the repository's standalone `bench_c8_vs_fp16_paged_310.py`. Inputs were
generated on CPU and copied to the NPU because the standalone container's
`StatelessRandomNormalV2` AICPU kernel is unavailable. Each path used five
warmups and twenty synchronized measurements. The test shape was GQA with
`query_heads=16`, `kv_heads=2`, `head_dim=256`, `block_size=64`, and
`splits=4`.

| Context | Native FP16 paged | Staged C8 + native PA | Direct C8 cube | Direct cube max error |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 414 us | 463 us | 1,161 us | 0.00106 |
| 32,768 | 1,134 us | 2,008 us | 8,101 us | 0.00035 |
| 114,688 | 3,216 us | 6,475 us | 27,886 us | 0.00017 |

All paths passed the numerical threshold against the FP16 reference. The
114,688-token graph replay for the staged path also passed with maximum error
`1.79e-05` against the quantized reference.

The result is a performance rejection of the current direct fused prototype,
not a correctness rejection. The cube kernel performs dequantization, QK,
online softmax, and V accumulation in Ascend C, but it repeatedly processes
small vector tiles and launches enough work that its latency grows much faster
than native paged attention. The staged path is closer to native performance,
but requires a complete FP16 NZ K/V temporary buffer, so it loses the memory
benefit of C8 and is still slower at every tested length.

Consequently this implementation remains an experimental reference and is not
enabled in the vLLM service. The next long-context optimization should target
the generated `paged_attention_decoder_mask_0` and the conversion/attention
interface, ideally by reusing native PA's optimized reduction and mask path or
by obtaining a hardware-supported fused primitive. Replacing native PA with
the current Ascend C cube kernel would regress both 4K and 112K workloads.

Reproduction (after stopping the service on the experimental NPU):

```bash
python3 experimental/bench_c8_vs_fp16_paged_310.py \
  --library /work/artifacts/c8-transdata-build/output/libqwen36_c8.so \
  --cube --splits 4 --staged --lengths 4096 32768 114688
```
