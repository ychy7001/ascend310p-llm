# WY ReduceSum runtime audit (2026-11-07)

## Scope

This was a read-only runtime observation of the frozen original Torch WY arithmetic. It used two existing rank-0 T640 captures and evaluated only the original FP32 `product.sum(-2)` at rows 3, 16, 31, and 63. It did not change the WY kernel, thresholds, worker, protocol, model, or production image, and it did not enter G1, G2, or performance acceptance.

## Findings

- All 8 observed outputs were byte-identical to the same Torch `sum(-2)` result.
- Every trace contained `aten::sum` and `aclnnReduceSum`.
- The runtime kernel identity was the same in all 8 observations:
  `ReduceSum_483c206567cda955ff1cdfd62744bc74_high_performance_110001000900`.
- The observed path is therefore `aten::sum -> aclnnReduceSum -> ReduceSum -> 310P high_performance kernel` for this FP32 shape and current CANN/Torch stack.
- The trace identifies the selected binary and tiling suffix, but does not expose enough reduction-instruction detail to prove the internal accumulation tree or make a byte-equivalent fused replacement safe.

## Runtime window

- Window: `wy-reducesum-runtime-20261107`
- Budget: 900 seconds; actual 16.50 seconds
- Exit code: 0
- Independent cleanup: passed; rootful/rootless containers empty, both services inactive, no NPU processes
- Summary SHA256: `d68d13ee4faf54a29fccfd8839254a07c3bcedf8629c09b89e388d9d9ec8ca46`
- Cleanup record SHA256: `ad6aba7e8cc816b71005db565bdafa7396bd8f21399332c94d82687b1ca19dfe`

## Decision

The actual ReduceSum entry point is now confirmed, so a targeted fusion prototype is technically possible. A direct sequential FP32 accumulator is not accepted as equivalent based on this evidence. The next step is static reconstruction of the selected `110001000900` schedule and its input/output layout, followed by a CPU byte-equivalence check before any new device window. No production adoption follows from this audit.

Frozen protocol, rules, worker, R, WY Torch module, and production kernel SHA values were unchanged.
