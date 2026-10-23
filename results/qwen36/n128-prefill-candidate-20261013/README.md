# N=128 Gate/Up Prefill Candidate

This candidate changes only the large-prefill dispatch for `nDim=1024` to
`Schedule<M=32, N=128>`. The down projection remains on the existing N=64
fallback, and Decode-specific dispatches are unchanged. It was compiled as a
complete three-op Ascend 310P OPP package.

Environment: one Ascend 310P3, TP1, W4A8, eager, MTP=0, FP16 KV,
`max_model_len=8192` for the initial 4096 candidate, then 12288 for 8192
candidate and both baseline tests; memory utilization 0.90, unique prompts, one warmup and
three measured requests.

For 4096 input tokens the valid Prefill measurements were:

```text
401.1729 / 400.6277 / 401.2060 tok/s
median: 401.1729 tok/s
```

The stable M=32/N=64 dispatch is approximately 375--380 tok/s for the same
workload family. The 5--7% difference is from short sequential measurements,
not an alternating A/B test.

At 8192 input tokens, Prefill was `397.3443 / 396.7822 / 396.1008 tok/s`
(median `396.7822`). At 8192 input/256 output, Prefill was
`397.9453 / 398.5922 / 397.8720 tok/s` (median `397.9453`) and Decode was
`21.2352 / 21.2181 / 21.1859 tok/s` (median `21.2181`). All requests passed
usage validation. Decode dispatch is unchanged, so this is a regression check
rather than a matched baseline comparison.

Five short generation sanity cases passed. The file named
`n128-prefill-candidate-needle-20261013.jsonl` contains a benchmark prompt and
14-token response only; it did not perform needle retrieval and must not be
cited as one. Treat this as an experimental candidate until a matched baseline
A/B run, actual retrieval checks, and validation with the product C8/MTP/Graph
configuration are complete.

Raw files in `results/` are `n128-prefill-candidate-20261013.jsonl` (4096),
`n128-prefill-candidate-20261013-8192.jsonl` (8192 prefill),
`n128-prefill-candidate-20261013-8192-o256.jsonl` (prefill and decode),
`n128-prefill-candidate-sanity-20261013.json`, and
`n128-prefill-candidate-needle-20261013.jsonl` (benchmark-only, not retrieval).

## Stable N=64 baseline repeat

The same server launcher, model, extension and image were used with the stable
three-op OPP, TP1/eager/MTP0/FP16 KV/.90. Request prompt IDs and phase protocol
match the candidate. This is one sequential candidate/baseline pair, not an
alternating or statistically powered result.

| Input/output | N=64 Prefill median | N=128 Prefill median | Difference |
| --- | ---: | ---: | ---: |
| 4096/1 | 376.2280 | 401.1729 | +6.63% |
| 8192/256 | 371.2599 | 397.9453 | +7.19% |

8192/256 Decode medians were 21.0576 (N=64) and 21.2181 tok/s (N=128),
about +0.76%; no material Decode improvement is claimed. Raw baseline files
are `n128-prefill-baseline-n64-20261013-4096.jsonl` and
`n128-prefill-baseline-n64-20261013-8192-o256.jsonl` in `results/`.
The native candidate source SHA256 is
`8727515fba385b37604faa35715d59d2374017d3105a05d71b99250f0f4e7350`.
