# Ascend 310P 可复用实验工具

通用工具接收显式输入，不绑定当轮容器名或结果日期。固定 PR、运行时源码哈希和数值门禁属于工具的适用契约，继续保留；整理没有修改算法、阈值或判定。

2026-11-03—04 的一次性控制器、固定样本诊断、wrapper、manifest 和补丁位于 [实验归档](../../experimental/runs-20261103-04/ARCHIVE.md)，原始操作记录一并保留。旧文档中指向已移动脚本的路径应按该归档定位。

`tools/check_business_quality.py` 必须通过 `--url` 或 `--urls` 显式传入端点，不再默认指向本轮端口。归档中的私有主机和用户目录已替换为占位符，通用 loopback 和监听地址保留。

以下只列入口，不执行设备任务。门禁和算子工具需要匹配的 Ascend 运行时、设备及制品；CPU 单元测试独立运行。

## 可复用清单（含 tools 改动，共 52 个脚本/测试）

- `benchmarks/ascend310p/analyze_wy_snapshots.py`
- `benchmarks/ascend310p/audit_runtime_stack.py`
- `benchmarks/ascend310p/build_fwd_o_maskfix_extension.py`
- `benchmarks/ascend310p/build_pr17640_fwd_o_extension.py`
- `benchmarks/ascend310p/build_pr17640_gdn_pair_extension.py`
- `benchmarks/ascend310p/build_pr17640_wy_extension.py`
- `benchmarks/ascend310p/check_action_state_regression.py`
- `benchmarks/ascend310p/check_cann921_gdn_pair.py`
- `benchmarks/ascend310p/check_default_gdn.py`
- `benchmarks/ascend310p/check_dense_format_diagnostic.py`
- `benchmarks/ascend310p/check_evidence_protocol.py`
- `benchmarks/ascend310p/check_meeting_prompt_policy.py`
- `benchmarks/ascend310p/check_native_wy.py`
- `benchmarks/ascend310p/check_native_wy921.py`
- `benchmarks/ascend310p/check_pr17640_gdn_pair.py`
- `benchmarks/ascend310p/check_short4_logprobs.py`
- `benchmarks/ascend310p/diagnose_native_wy.py`
- `benchmarks/ascend310p/download-pinned-huggingface.py`
- `benchmarks/ascend310p/download-pinned-modelscope.py`
- `benchmarks/ascend310p/export-production-profiles.py`
- `benchmarks/ascend310p/gpu_scheduler_observer.py`
- `benchmarks/ascend310p/inventory-checkpoint.py`
- `benchmarks/ascend310p/isolate_wy_stages.py`
- `benchmarks/ascend310p/native_wy.py`
- `benchmarks/ascend310p/probe_dense_quant_matmul.py`
- `benchmarks/ascend310p/probe_fwd_o_frozen.py`
- `benchmarks/ascend310p/probe_gdn_call_context.py`
- `benchmarks/ascend310p/probe_gdn_intermediates.py`
- `benchmarks/ascend310p/recheck_meeting_owners.py`
- `benchmarks/ascend310p/replay_dense_quant_capture.py`
- `benchmarks/ascend310p/repro_chunk_gdn_nondeterminism.py`
- `benchmarks/ascend310p/resolve_evidence_owners.py`
- `benchmarks/ascend310p/sample-resources.py`
- `benchmarks/ascend310p/summarize.py`
- `benchmarks/ascend310p/summarize_action_state_pair.py`
- `benchmarks/ascend310p/test_summarize.py`
- `tools/analyze_decode_kernels.py`
- `tools/bench_http_stream.py`
- `tools/bench_prefix_reuse.py`
- `tools/bench_server_phases.py`
- `tools/check_business_facts.py`
- `tools/check_business_quality.py`
- `tools/check_prefix_state.py`
- `tools/profile_http_decode.py`
- `tools/profile_http_decode_full.py`
- `tools/profile_http_prefill.py`
- `tools/replay_stage_workload.py`
- `tools/test_analyze_decode_kernels.py`
- `tools/test_bench_http_stream.py`
- `tools/test_bench_prefix_reuse.py`
- `tools/test_check_business_facts.py`
- `tools/test_replay_stage_workload.py`

## 本地验证

```bash
PYTHONPATH=tools:benchmarks/ascend310p python3 -m unittest discover -s tools -p 'test_*.py'
PYTHONPATH=tools:benchmarks/ascend310p python3 -m unittest discover -s benchmarks/ascend310p -p 'test_*.py'
```
