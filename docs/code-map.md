# 代码入口与维护边界

## 构建、模型与运行

| 目录/入口 | 职责 | 维护原则 |
| --- | --- | --- |
| scripts/build-native.sh | 固定社区基线、覆盖必要算子快照、构建独立OPP/扩展 | 不整体合并fork；固定SDK/镜像 |
| native-overlay/csrc/gmm/ | W4 pack、MatMul、SwiGLU及相关源码 | 数值/shape门禁后调整tile；候选不要自动默认启用 |
| tools/qwen36_int4_custom.cpp | 独立Torch算子namespace | 保持官方_C_ascend不冲突 |
| runtime/.../qwen36_w4_patch.py | routed MoE构造/加载、TP所有权、MTP排除 | 按量化metadata opt-in；后续替换临时类monkeypatch |
| runtime/.../qwen4_exp/ | 社区W4执行子集与ModelSlim ABI适配 | 保留来源；裁剪前先核实依赖 |
| bootstrap/ | 隔离进程加载适配层与fake contracts | 不修改宿主安装 |
| tools/convert_modelslim_w4a8.py等 | BF16→专家sidecar→自包含checkpoint | 不覆盖源权重；保留哈希/元数据 |
| scripts/run-service.sh | W4/FP16 KV基线启动 | 与C8实验启动器分开 |

## C8与测试

| 入口 | 职责 |
| --- | --- |
| experimental/c8_vllm_adapter.py | cache spec、预算、allocate及full-attention读取接入 |
| experimental/c8_cache_310.py、ascend_c8_kernel/ | INT8持久cache、FP16 scale、量化写入/读取算子 |
| experimental/direct_graph_310.py | 固定镜像Graph兼容诊断；尚非正式上游修复 |
| experimental/run-c8-112k-product.sh | 112K显式mtp1/mtp2候选档 |
| experimental/check-n128-product.sh | 复用、阶段计时和实际112K检索初步门禁 |
| experimental/run-w4-prefill-check.sh | 独占NPU的真实专家矩阵独立检查 |
| tools/bench_server_phases.py、compare_server_phases.py | 独占端点阶段计时/协议对比；记录输入哈希 |
| tools/eval_gsm8k_http.py、compare_gsm8k_runs.py | 固定子集逐题质量与截断比较 |
| results/ | 后续原始数据、关键日志与实验README |

当前整理没有移动运行入口或大规模改写已验证kernel，避免破坏复现命令。独立探针保留在experimental/，不加载到普通服务。下一步减小主线改动范围的具体清单见[会议评审报告](huawei-adaptation-review-20261013.md)。

## 源码与制品追踪

首发历史清单为`docs/runtime-source-sha256.json`；后续2026-10-13核心运行/构建快照为`docs/source-snapshot-20261013.json`。清单只校验源码，不证明二进制由该源码构建或通过NPU测试；二进制哈希与实际日志仍需逐实验核对。

```bash
python3 tools/check_source_snapshot.py docs/source-snapshot-20261013.json
```

新清单须随被追踪文件的修改同步维护，并通过上述命令；不要重写首发历史清单。Git记录说明每轮实际修改与测量依据。代码来源及许可证见[provenance](provenance.md)和NOTICE。
