# WY ReduceSum runtime audit (2026-11-07)

## Scope

This was a read-only runtime observation of the frozen original Torch WY arithmetic. It used two existing rank-0 T640 captures and evaluated only the original FP32 `product.sum(-2)` at rows 3, 16, 31, and 63. It did not change the WY kernel, thresholds, worker, protocol, model, or production image, and it did not enter G1, G2, or performance acceptance.

## Findings

The first probe in commit `[18]` was invalid: it selected `base[:, :, row, :row]` and reduced that tensor, which observed a `[1, 16, row]` head-axis reduction rather than the WY expression `(row.unsqueeze(-1) * sub).sum(-2)`. Its reported `110001000900` kernel must not be used as WY evidence. The invalid window is retained for auditability and superseded below.

The corrected probe reconstructed the original row recurrence and then profiled the actual WY product reduction.

- All 8 observed outputs were byte-identical to the same Torch `sum(-2)` result.
- Every trace contained `aten::sum` and `aclnnReduceSum`.
- The corrected runtime kernel identity was the same in all 8 observations:
  `ReduceSum_483c206567cda955ff1cdfd62744bc74_high_performance_10000000900`.
- The tiling key decodes to non-atomic reduction, double buffering, pattern 9, block split axis 0, and UB split axis 0. The installed scheduler associates this non-last-axis accumulation path with `dichotomy_reduce`, `reuse_src_tensor`, and `nlast_reduce_dichotomy=16`.
- Several CPU implementations of sequential, pairwise, and 16-wide grouped reduction were compared against the saved NPU outputs. Rows 3 and 16 matched some candidates; rows 31 and 63 matched none byte-for-byte, with maximum absolute differences from `1.49e-8` to `2.98e-8`.
- The observed path is therefore confirmed as `aten::sum -> aclnnReduceSum -> ReduceSum -> 310P high_performance kernel`, but its exact internal reduction tree remains unproven. A fused replacement is not safe to adopt.

## Runtime window

- Corrected window: `wy-reducesum-runtime-20261107-v3`
- Budget: 900 seconds; actual 16.50 seconds
- Exit code: 0
- Independent cleanup: passed; rootful/rootless containers empty, both services inactive, no NPU processes
- Summary SHA256: `0398c3150c6bc2015519fa1342fa0f8eb08795331e034d535303ab04f451fd5f`
- Cleanup record SHA256: `232800283fe7a430950928761048a296100172a284be5a12351859a6b4fb95c4`

## Decision

The actual WY ReduceSum entry point and accumulation class are now confirmed, so a targeted fusion prototype is technically possible but not yet justified. The next step is to inspect the compiled kernel or obtain a source-level implementation of the `dichotomy_reduce` tree; another hand-written sequential or simple pairwise CPU tree is insufficient. No production adoption follows from this audit.

Frozen protocol, rules, worker, R, WY Torch module, and production kernel SHA values were unchanged.
