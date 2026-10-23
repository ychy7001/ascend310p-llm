# Ascend 310P LLM Inference Adaptation

English | [中文](README.md)

Inference adaptation for large MoE models on Atlas 300I DUO (Ascend 310P3): low-bit MoE operators, speculative decoding (MTP), INT8 KV cache (C8), decode graphs, and long-context validation. Every claim links to raw data, and verified results are kept separate from experimental ones.

> Not an official Huawei, Qwen or vLLM release. The code builds on the community [opensensor/vllm-ascend](https://github.com/opensensor/vllm-ascend) 310P W4 work; see [provenance](docs/provenance.md) and [NOTICE](NOTICE). Detailed docs are currently written in Chinese.

## Status

| Model | Configuration | Status | Entry |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | W4A8 + BF16 MTP, TP1, decode graph, 112K | Passed startup, generation and long-context retrieval on real 310P3 | [qwen36](docs/qwen36/README.md) |
| Qwen3.6-35B-A3B | C8 (INT8 KV) | Experimental, relies on an updatable-graph workaround | [C8](docs/qwen36/c8.md) |
| Qwen3.6-35B-A3B | TP2 C8, batched concurrency, multimodal, formal P95 | **Not validated** | — |
| Qwen3.8-Flash-Next | W4A8 + C16, TP2/EP2 (two 310P), MTP0, decode graph | Validated up to 16K; a 32K request could not be scheduled | [qwen38](docs/qwen38/README.md) |
| Qwen3.8-Flash-Next | MTP, C8, concurrency, full quality eval | **Not validated** | — |

## Weights

| Artifact | Location | Status |
| --- | --- | --- |
| Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P | [ModelScope](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P) | Published; all 64 files checked by SHA256 ([record](docs/qwen36/release-validation.md)) |
| Qwen3.8 | Uses community [matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i); no weights republished here | — |

## Representative results (single stream, fixed protocol, not P95)

- Qwen3.6 MTP2 vs MTP1 (112K graph, 32K in / 256 out): decode 23.20 → 28.94 tok/s (~+24.8%), prefill ~−0.8%.
- Qwen3.6 MindIEQ long-prefill routing vs stock W4 (median of five alternating runs): 8K +23.29%, 32K +20.69%, decode change < 0.5%.
- Qwen3.8 TP2 decode graph (512/128, median of three): 13.87 → 16.67 tok/s (~+20%).
- Qwen3.8 TP2: fixed a causal-mask bug in the community GDN prefill kernel (wrong output at token 56 for T≥64): [patch](patches/qwen38/gdn-causal-mask-vector.patch), [write-up](results/qwen38/tp2-serving-20261018/README.md).

## Who did what

| Part | Origin |
| --- | --- |
| 310P W4 base operators, qwen4_exp model code | Community (matteius / opensensor), original copyright retained |
| Qwen3.6 adaptation, ModelSlim conversion, BF16 MTP, weight release | This project |
| C8 paged-KV operators and vLLM integration | This project (experimental) |
| MindIEQ prefill routing, A8 quantization | This project; external matrix classes stay under the CANN license and are not vendored |
| GDN state-row bug diagnosis and fix | This project; upstream [vllm-ascend#18730](https://github.com/vllm-project/vllm-ascend/pull/18730) (under review), issue #17856 |
| GDN WY/FwdH/FwdO rewrite | Upstream PR #17640 (others) |
| Qwen3.8 TP2 capacity audit, GDN mask fix, budget patch, graph validation | This project, on pinned community forks |

## Layout

`docs/qwen36`, `docs/qwen38`, `results/qwen36`, `results/qwen38` hold per-model docs and raw data. Shared code lives in `tools/`, `scripts/`, `experimental/`, `native-overlay/`, `runtime/`, `bootstrap/`, `deploy/`, `patches/`. Earlier paths remain reachable through Git history.

## Known limitations

- The graph path uses an experimental updatable-graph switch, not an upstream default.
- The public `build-native.sh` has not been run verbatim end to end in a fresh environment.
- Performance numbers are fixed-input single-stream; quality checks are small synthetic or closed-form probes, not full accuracy evaluation.
- Internal delivery and site-specific files were removed from HEAD; they remain in Git history.

License: Apache-2.0. Third-party licenses: see NOTICE.
