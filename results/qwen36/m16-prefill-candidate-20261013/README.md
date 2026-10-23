# Rejected M=16 Prefill Candidate

This experiment changed only the prefill dispatch for `4096 <= numRows <=
16384` from the stable M=32 schedule to M=16. It was compiled as a complete
three-op OPP package (MatMul, Pack, and SwiGLU Pack) and loaded in an isolated
TP1/eager/MTP=0/FP16-KV service on one Ascend 310P3.

The three unique 4096-token requests were valid and produced:

```text
329.0789 tok/s
328.7223 tok/s
328.8313 tok/s
median: 328.8313 tok/s
```

The matched stable M=32 schedule is approximately 375--380 tok/s for the
same workload family. M=16 therefore regresses by about 13%, consistent with
the extra tile and synchronization work. The dispatch change was reverted;
the repository source remains on the stable M=32 path.

The raw request log is `m16-prefill-4096.jsonl`. The earlier M=64 candidate is
recorded in `results/qwen36/m64-prefill-candidate-20261013` and was also rejected.
