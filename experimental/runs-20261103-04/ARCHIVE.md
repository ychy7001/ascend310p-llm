# 2026-11-03—04 实验脚本归档

本目录保留当轮控制器、服务 wrapper、固定样本诊断、生产命令 manifest 和未验证补丁。原始位置均为 `benchmarks/ascend310p/`，相对文件名保持不变。仅移动并替换私有主机及用户目录为占位符；算法、阈值、判定与执行顺序保持原样。

`README.md` 是当轮原始操作记录，其中路径和执行说明属于历史上下文。占位符 `LAB_PRIMARY`、`LAB_REPLICA`、`GPU_HOST`、`GPU_SERVICE` 和 `/path/to/` 需要显式填写。通用 loopback 地址、监听地址及端口保留。

通用工具留在 `tools/` 和 `benchmarks/ascend310p/`。归档脚本的相邻文件引用、仓库相对路径与远端目录布局未重写；追溯执行时须按原记录在独立工作目录放置所需工具和制品。归档文件不作为可直接运行的通用入口。原始结果只保留在本地，仓库中的 README/adjudication 是脱敏摘要；原始制品摘要不因脱敏而重算。

## 一次性脚本（47）

- `build-cann921-native-wy.py`
- `build-native-wy.py`
- `build-validate-fwd-o-maskfix.py`
- `build-validate-gdn-pair.py`
- `build-validate-upstream-fwd-o.py`
- `cann921_gdn_pair_worker.py`
- `check_maskfix_gdn.py`
- `check_wy_cpu_oracle.py`
- `gdn_pair_worker.py`
- `launch-frozen.sh`
- `launch-production-replica.py`
- `maskfix_worker.py`
- `native_wy_worker.py`
- `probe_native_wy_order.py`
- `quant_repack_worker.py`
- `quant_trace_worker.py`
- `recheck-native-wy-batching.py`
- `recheck-native-wy-control.py`
- `resume-production-baseline.py`
- `run-action-state-paired.py`
- `run-baseline.py`
- `run-cann921-gdn-focused.py`
- `run-cann921-model-screen.py`
- `run-cann921-operator-screen.py`
- `run-dense-baseline.py`
- `run-dense-quant-captured-replay.py`
- `run-dense-quant-isolation.py`
- `run-executor-cache-control.py`
- `run-fwd-o-isolation.py`
- `run-fwd-o-rebuild-control.py`
- `run-gdn-pair-screen.py`
- `run-gdn-runtime-control.py`
- `run-graph14-screen.py`
- `run-local-gpu35b-screen.py`
- `run-native-wy-screen.py`
- `run-native-wy921-gate.py`
- `run-production-baseline.py`
- `run-profile-stage.py`
- `start-cann921-gdn-pair.sh`
- `start-cann921.sh`
- `start-experiment.sh`
- `start-gdn-pair.sh`
- `start-maskfix.sh`
- `start-native-wy.sh`
- `start-production-replica.sh`
- `start-quant-trace.sh`
- `validate-native-wy.py`
