# Ascend 310P LLM Inference Adaptation

[English](README.en.md) | 中文

在 Atlas 300I DUO（Ascend 310P3）上运行大模型的推理适配工程：低比特 MoE 算子、推测解码（MTP）、INT8 KV（C8）、Decode Graph 与长上下文验证。所有结论附原始数据，并明确区分「已验证」与「实验/未验证」。

> 本项目不是华为、Qwen 或 vLLM 官方发布。源码基于社区 [opensensor/vllm-ascend](https://github.com/opensensor/vllm-ascend) 的 310P W4 实现，见 [来源与署名](docs/provenance.md) 与 [NOTICE](NOTICE)。

适配过程的叙述版本写在知乎专栏[昇腾 310P 推理适配笔记](https://www.zhihu.com/column/c_2097812513450734740)，按容量核算、启动排障、算子 bug 定位分篇；仓库这边是原始数据和复现入口。

## 模型状态

| 模型 | 配置 | 状态 | 入口 |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | W4A8 + BF16 MTP，TP1，Decode Graph，112K | 已在真实 310P3 通过启动、生成、长检索门禁 | [qwen36](docs/qwen36/README.md) |
| Qwen3.6-35B-A3B | C8（INT8 KV） | 实验后端，依赖 updatable-graph 补丁 | [C8](docs/qwen36/c8.md) |
| Qwen3.6-35B-A3B | TP2 C8、并发批处理、多模态、正式 P95 | **未验收** | — |
| Qwen3.8-Flash-Next | W4A8 + C16，TP2/EP2（双 310P），MTP0，Decode Graph | 已验证到 16K；32K 请求无法调度 | [qwen38](docs/qwen38/README.md) |
| Qwen3.8-Flash-Next | MTP、C8、并发、完整质量评估 | **未验证** | — |

## 权重

| 制品 | 位置 | 状态 |
| --- | --- | --- |
| Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P | [ModelScope](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P) | 已发布，64 文件逐一 SHA256 校验，见 [release-validation](docs/qwen36/release-validation.md) |
| Qwen3.8 | 使用社区 [matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i)（Qwen Community License 1.0），本项目未转载权重，下载与许可见 [qwen38](docs/qwen38/README.md) | — |

## 代表性结果（均为单流、固定协议，非 P95）

- Qwen3.6 MTP2 vs MTP1（112K Graph，32K 输入 / 256 输出）：Decode 23.20 → 28.94 tok/s（约 +24.8%），Prefill 约 −0.8%。
- Qwen3.6 MindIEQ 长 Prefill 分流 vs 原 W4（交替五次中位）：8K +23.29%、32K +20.69%，Decode 变化 < 0.5%。
- Qwen3.8 TP2 Decode Graph（512/128，三次中位）：13.87 → 16.67 tok/s（约 +20%）。
- Qwen3.8 TP2 修复了社区 GDN Prefill 因果掩码错误（T≥64 时 token 56 输出越界），见 [补丁](patches/qwen38/gdn-causal-mask-vector.patch) 与 [记录](results/qwen38/tp2-serving-20261018/README.md)。

## 我做了什么 / 社区做了什么

| 部分 | 来源 |
| --- | --- |
| 310P W4 基础算子与 qwen4_exp 模型实现 | 社区（matteius / opensensor），保留原版权 |
| Qwen3.6 适配、ModelSlim 量化转换、BF16 MTP、权重发布 | 本项目 |
| C8 分页 KV 算子与 vLLM 接入 | 本项目（实验） |
| MindIEQ Prefill 分流、A8 量化、native 布局 | 本项目，外部矩阵类遵循 CANN 许可，不入库 |
| GDN 状态行 BUG 定位与修复 | 本项目，上游 [vllm-ascend#18730](https://github.com/vllm-project/vllm-ascend/pull/18730)（审阅中）/ issue #17856 |
| GDN WY/FwdH/FwdO 重写 | 上游 PR #17640（他人） |
| Qwen3.8 TP2 容量审计、GDN 掩码修复、预算补丁、Graph 验证 | 本项目，基于社区 fork 固定提交 |

## 目录

```
docs/            共享文档（来源、代码地图）
docs/qwen36/     Qwen3.6 文档（含模型卡、验证、C8、部署）
docs/qwen38/     Qwen3.8 文档
results/qwen36/  Qwen3.6 原始数据与实验记录
results/qwen38/  Qwen3.8 原始数据与实验记录
tools/ scripts/ experimental/ native-overlay/ runtime/ bootstrap/ deploy/ patches/
```

结果目录名带日期，不改写；旧版本 commit 中的路径仍可通过 Git 历史访问。

## 复现

构建、启动、校验见 [Qwen3.6 文档](docs/qwen36/README.md)。需要 Ascend 310P 硬件、CANN 9.1 与驱动（不随仓库分发）。

## 已知限制

- Graph 路径使用实验性 updatable-graph 开关，不是上游默认实现。
- 公开 `build-native.sh` 未在全新环境逐字端到端验收（见 release-validation）。
- 所有性能数据为固定输入、单流；质量检查为小规模合成/闭式答案，不代表完整准确率评测。
- 内部交付、会议材料和站点部署文件已移出主树，仍可在 Git 历史查阅。

License: Apache-2.0。第三方许可见 NOTICE。
