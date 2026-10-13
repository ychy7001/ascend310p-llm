# N=64 matched product-combination baseline

Date: 2026-10-13. This is a sequential A/B comparison against the N=128
candidate in `results/n128-product-20261013/`, on a2's first Ascend 310P3.
Both services used TP1, W4A8, C8, MTP2, FULL_DECODE_ONLY Graph, mode 0,
capture sizes `[1,2,3]`, `max_model_len=114688`, memory utilization `.95`,
batch 1024, expert chunk 512, prefix cache off, and the same model, C8
library, extension, container image and launcher. Only the routed gate/up
Prefill OPP was changed from N=64 to the N=128 candidate. The services were
never concurrent.

| Input/output | N=64 Prefill | N=128 Prefill | Change | N=64 Decode | N=128 Decode | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192/128 | 364.4936 | 391.5909 | +7.43% | 37.1222 | 37.0124 | -0.30% |
| 32256/128 | 335.7245 | 357.5768 | +6.51% | 29.0776 | 29.0453 | -0.11% |

Each value is the median of three unique-prompt requests after one warmup;
phase counters and usage were valid. The source benchmark files use the same
warmup prompt configuration, but were produced before measured prompt hashes
were added, so exact measured prompt equality is not claimed. These are
sequential pairs, not P95 or randomized alternating measurements.

Both services passed strict answer checks (8/8), A/B/A cache-reuse checks,
and a 114560-token retrieval request. N=64 took 435.7018 seconds and returned
`ASCEND732941`; N=128 took 415.3063 seconds. A single long retrieval is a
capacity/correctness gate, not an end-to-end performance or accuracy score.

The N=128 result supports retaining the candidate for further business and
tail-latency tests: the observed benefit is Prefill-only, while Decode is
within measurement noise. It does not justify a broad speedup claim.
