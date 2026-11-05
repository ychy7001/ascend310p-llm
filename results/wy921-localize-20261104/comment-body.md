Thanks for the work on the 310P GDN path. Here is a single-chunk/head runtime WY input with localized numerical exceedances at `18163f6b1d208ab3b4fb06e9752225611c37f40e`.

Environment recorded on 2026-11-04: Atlas 300I Duo / 310P3, CANN 9.2.1, torch 2.10.0, torch-npu 2.10.0.post7. Input: B=1, T=64, Hk=Hv=1, K=V=128; contiguous q/k/v/beta FP16, g FP32. No weights, tokenizer, state, varlen metadata or TP communication are needed. The input combines correlated keys (max off-diagonal |k_i·k_j| ≈ 0.999973), max |v| = 18.078125, weak decay (sum(g) ≈ -0.0454993), and beta in [0.467773, 0.963867].

Expected under the existing WY gate: exact q/k; for other outputs, every element within `atol=5e-4, rtol=0.04`, plus relative RMSE ≤ 0.02. Recorded u results (8192 elements):

| Comparison | Max absolute difference | Elements outside pointwise tolerance |
| --- | ---: | ---: |
| Native vs CPU FP32 triangular solve | 0.0039176941 | 1 |
| Native vs Torch WY | 0.01171875 | 2 |
| Torch WY vs CPU FP32 | 0.0122814178 | 2 |

The maximum difference and the out-of-tolerance elements can be at different coordinates. q/k/w/g passed; native repetition was bitwise equal. Torch WY also has local exceedances, so disagreement with Torch alone does not establish native incorrectness. In recorded controls, scaling v down 10×, setting g=-1, or replacing k with random normalized vectors brought native-vs-CPU within tolerance.

[Reproduction files and full build instructions](https://gist.github.com/ychy7001/b94596c3949f19012d635f97b203dd23): text-only scripts, lossless numeric input, recorded w/u/g outputs and SHA256SUMS. Clone the gist, then:

1. Run `sha256sum -c SHA256SUMS`, `python3 -B -m unittest -v test_wy_snapshot_reference.py`, and `python3 -B validate_cpu.py` on CPU.
2. In a CANN 9.2.1 environment, follow the README to check out the pinned PR, build only `chunk_gated_delta_rule_compute_wy` and its standalone binding, then run `repro-real-wy.py` with the documented input/source/library/vendor arguments. The device script exits 1 when the comparison gate fails and writes all three comparisons.

The CPU tests, lossless input reconstruction and recomputation from recorded device outputs passed during review (torch 2.5.1+cpu). The device build/run was not repeated during this review; the exact original driver version was not retained. This isolates an observed numerical case, without a unique root-cause claim, and has **not** been shown to fully explain the model-level logprob discrepancy.
