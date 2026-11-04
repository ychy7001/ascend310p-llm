# Draft: Ascend 310P chunk GDN output changes across identical calls

Status: evidence draft; not posted. The root cause and a fix are not established.

On a two-card Ascend 310P development host, repeated calls to the installed production `chunk_gated_delta_rule_310` can produce different output values for identical inputs, while the final recurrent state remains bitwise stable. The reproduction uses synthetic tensors only; no model weights, HTTP server, native WY extension, or model benchmark is required.

Environment frozen for the reproduction:

- vLLM commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- vLLM-Ascend commit `88eac2713b14beedc7a48803274f8d6cee3e9f30`.
- torch `2.10.0+cpu`, torch-npu `2.10.0.post2`, CANN `9.1.0-beta.1`; host driver `26.1.1`.
- Production chunk module SHA256 `ab2b739bb6630ebe9fcb1d7af820f93adffce79668e32247bbb869119a0d6c69`.
- Production FwdO/FwdH/common source: 59 files verified against the production commit Git blobs.

Use `benchmarks/ascend310p/repro_chunk_gdn_nondeterminism.py` in the matching environment with custom operators enabled:

```bash
python3 repro_chunk_gdn_nondeterminism.py --out original-repeatability.jsonl --repeats 48
```

The script sets seed 311 and constructs FP16 normalized Q/K, FP16 V and beta, FP32 non-positive g, and a nonzero FP16 initial state. Shapes: B=1, T=111, Hk=8, Hv=16, K=V=128. Its independent CPU reference applies the token recurrence directly; it does not reuse chunk/WY/FwdH/FwdO computations. Elementwise tolerance is `atol=2e-4, rtol=0.04`, with an additional relative RMSE limit of 0.02. Per-call raw metrics are written to JSONL; the diagnostic script does not claim a fix.

Observed evidence from separate executions (rates are run-specific):

| Probe | Observation |
| --- | --- |
| CPU production chunk algebra vs independent CPU recurrence | max absolute output error 3.59e-6, state error 5.22e-8 |
| Standalone production reproduction without a native WY extension | 5/12 calls inside tolerance; deviation at token 56; state bitwise stable |
| Original full call order, interface capture after final output | WY outputs, FwdH outputs and all FwdO inputs bitwise stable; raw FwdO output deviation matches the final transposed output |
| Frozen FwdO inputs, repeated FwdO alone | 48/48 inside tolerance and bitwise stable |
| Original FwdO source independently rebuilt, API renamed only | 48/48 inside tolerance and bitwise stable in the same frozen-input probe |
| Independently rebuilt original FwdO, full chain | 18/24 outside tolerance with Torch WY call history; 0/24 in the frozen-WY-clone phase of this run |

Adding host synchronization, cloning intermediate tensors, or changing earlier allocations can change the occurrence rate. Therefore the frozen-input probe alone does not establish correctness of the full chain. The current evidence narrows the observable deviation to raw FwdO output, but does not establish whether the root cause is ordering, memory lifetime, synchronization inside the kernel, or another runtime interaction.

Two local mask experiments (vector-only initialization and V/S fences around the original scalar tail updates) failed fixed-input checks, 0/48 each, against 48/48 for the paired production operator. Both are rejected; neither is an upstream fix proposal. The unchanged-source rebuild control passes the same probe, so a general incompatibility of that rebuild framework does not explain those two failures.

The existing GruntovDima PR17640 implementation is a separate upstream contribution. Our WY-only reuse improves Prefill but does not eliminate the repeatability issue across call histories. It must not be presented as our kernel implementation or as a confirmed numerical fix. End-to-end meeting-owner errors also occur on the unmodified baseline; no causal link to this synthetic case has been established.

Local evidence: `results/production-wy-batching-recheck-20261103/`, `results/fwd-o-isolation-20261103/`, `results/fwd-o-rebuild-control-20261103/`. A portable issue attachment should include the standalone script, exact environment metadata and raw JSONL, while excluding proprietary prompts and model weights.


## Cross-runtime control and contribution scope (2026-11-04)

The unchanged production Python chunk implementation was also checked in a second, separately audited complete image (torch-npu post4 / CANN 9.1.0). The standalone recurrence probe had 21/48 calls inside tolerance in the production post2 image and 16/48 in the post4 image. This is a whole-image compatibility screen, not an isolated torch-npu comparison or a statistically meaningful failure-rate comparison.

In full-call-chain capture, Torch-WY runs outside tolerance were 15/24 and 14/24 respectively. Frozen-WY-clone runs were 1/24 outside tolerance in each image, despite stable WY/FwdH outputs and FwdO inputs. Cloning is therefore not an established fix. Raw outputs matched the final CPU-transposed outputs. See `results/gdn-runtime-control-20261104/`.

Existing native-WY implementations and integrations include PR17640 and PR18057; metadata and integration differences are frozen in `results/upstream-gdn-pr-audit-20261104/`. Our experiment directly reuses the credited PR17640 implementation. No independent native-WY feature PR is proposed. Potential contributions are reproducible correctness tests and independently established fixes. Complete paired upstream FwdH/FwdO validation is in progress; the earlier rejected new-FwdO/old-FwdH combination did not test its NZ state exchange contract.

## 2026-11-04：完整H/O候选本轮最终判定

配套H/O+原生WY算子验证通过（四组192次输出检查、两次严格10例），模型质量筛选已结束。前后短请求各8/8，事实16/16，业务自动28/28；人工会议4仍有负责人错配及完成状态编造，因此最终not_accepted_business_semantics，不进入性能或部署。GSM64=63/64，与冻结生产基线逐样本答案/正确性一致，唯一index205截断；文本19/64完全一致，不证明逐token等价。A′此前也有业务错误，不能据此将错误归因为候选独有回归。

结果：results/production-gdn-pair-quality-20261104/adjudication.json。上游路线和补充研究取舍：docs/research/gdn-upstream-pr-evaluation.md。此前段落中的“运行中”是过程状态，以本段最终判定为准。

## 2026-11-04：CANN9.2.1完整默认栈隔离验证

用户同意后在a2拉取官方v0.30.0-310p（IDd103c59b31c6…），核实Toolkit/OPP/NNAL9.2.1、torch-npu2.10post7、vLLM/Ascend0.30.0；a1和宿主驱动固件未改。初始GDN T111两卡96/96、Dense无权重Matmul10/10通过，但扩展门禁13/15，两组T512输出偏差。独立进程T512 HV16仅29/48在容差内、HV24为0/48，末state各48/48稳定。最终not_accepted_gdn_correctness；未加载模型、未测性能，不宣称CANN单独修复Dense或GDN。

完整评估docs/research/cann-upgrade-evaluation.md；原始results/cann921-upgrade-20261104；可公开合成复现results/upstream-gdn-cann921-repro-20261104.tar.gz（尚未发布）。本轮至此保留隔离新版环境，下一步优先GDN输出正确性，不扩大WY重复实现。
