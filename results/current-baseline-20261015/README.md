# Current stable service baseline

Measured on A2 device 0 after restoring the ordinary service. Configuration:
TP1, W4A8, C8 device-length, MTP1, Decode Graph, prefix cache enabled,
`MAX_MODEL_LEN=114688`, `KV_BYTES=6442450944`, MindIEQ/GDN enabled, and one
exclusive request. Each length used three unique prompts, one warmup, output
128, speculative counters, and prefix-cache counters. All rows were valid;
prefix-cache hits were zero.

| Prompt/output | Prefill tok/s | Decode tok/s | MTP acceptance |
| ---: | ---: | ---: | ---: |
| 8,192 / 128 | 564.84 median (562.49--565.17) | 23.98 median (23.93--24.02) | 100% |
| 32,256 / 128 | 511.49 median (508.87--513.03) | 19.35 median (19.11--19.44) | 100% |

The 32K request is about 9.4% slower in Prefill and 20.3% slower in Decode
than the 8K request. These are current service phase-counter measurements, not
pure kernel timings or P95 results. Raw request records are in the two JSONL
files in this directory.
