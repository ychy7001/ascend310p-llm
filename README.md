# Ascend 310P LLM Inference Adaptation

[English](README.en.md) | 中文

本仓库维护 Atlas 300I DUO（Ascend 310P3）内部推理引擎的交付镜像、源码与验收记录：低比特 MoE 算子、推测解码（MTP）、INT8 KV（C8）、Decode Graph 与长上下文验证。所有结论附原始数据，并明确区分「已验证」与「实验/未验证」。

> 本项目不是华为、Qwen 或 vLLM 官方发布。源码基于社区 [opensensor/vllm-ascend](https://github.com/opensensor/vllm-ascend) 的 310P W4 实现，见 [来源与署名](docs/provenance.md) 与 [NOTICE](NOTICE)。

适配过程的叙述版本写在知乎专栏[昇腾 310P 推理适配笔记](https://www.zhihu.com/column/c_2097812513450734740)，按容量核算、启动排障、算子 bug 定位分篇；仓库这边是原始数据和复现入口。

## 模型状态

**双 310P 默认交付：Qwen3.6-35B-A3B-W8A8 + native WY + MTP2 + eager mask 复用。** [默认镜像、权重、验收与回退](docs/qwen36/w8a8-wy-delivery.md)。单卡使用 W4A8 TP1。

| 模型 | 配置 | 适用场景 | 状态 | 入口 |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | W8A8 + native WY + MTP2 + 主模型 Graph + eager mask 复用，TP2，262144 上下文，调度并发上限 4 | 双卡默认交付 | registry 已发布并验证；32K 冷 Prefill +7.55%，540 次真实算子 FP16 逐位一致 | [W8A8 262K 复现](docs/qwen36/w8a8-262k.md) |
| Qwen3.6-35B-A3B | W4A8 + BF16 MTP，TP1，Decode Graph，112K | 单卡档；双卡资源不足时的降级档 | 已在真实 310P3 通过启动、生成、长检索门禁 | [qwen36](docs/qwen36/README.md) |
| Qwen3.6-35B-A3B | C8（INT8 KV） | 实验后端 | 依赖 updatable-graph 补丁 | [C8](docs/qwen36/c8.md) |
| Qwen3.6-35B-A3B | TP2 C8、并发批处理、多模态、正式 P95 | — | **未验收** | — |
| Qwen3.8-Flash-Next | W4A8 + C16，TP2/EP2（双 310P），MTP0，Decode Graph | — | 已验证到 16K；32K 请求无法调度 | [qwen38](docs/qwen38/README.md) |
| Qwen3.8-Flash-Next | MTP、C8、并发、完整质量评估 | — | **未验证** | — |

## 权重

| 制品 | 位置 | 状态 |
| --- | --- | --- |
| Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P | [ModelScope](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P) | 已发布，64 文件逐一 SHA256 校验，见 [release-validation](docs/qwen36/release-validation.md) |
| Qwen3.8 | 使用社区 [matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i)（Qwen Community License 1.0），本项目未转载权重，下载与许可见 [qwen38](docs/qwen38/README.md) | — |

## 代表性结果（均为单流、固定协议，非 P95）

- Qwen3.6 MTP2 vs MTP1（112K Graph，32K 输入 / 256 输出）：Decode 23.20 → 28.94 tok/s（约 +24.8%），Prefill 约 −0.8%。
- Qwen3.6 MindIEQ 长 Prefill 分流 vs 原 W4（交替五次中位）：8K +23.29%、32K +20.69%，Decode 变化 < 0.5%。
- Qwen3.8 TP2 Decode Graph（512/128，三次中位）：13.87 → 16.67 tok/s（约 +20%）。
- Qwen3.8 TP2 修复了社区 GDN Prefill 因果掩码错误（T≥64 时 token 56 输出越界），见 [定位与修复](docs/qwen38/gdn-causal-mask-bug.md)、[补丁](patches/qwen38/gdn-causal-mask-vector.patch) 与 [原始记录](results/qwen38/tp2-serving-20261018/README.md)。

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
docs/qwen36/     Qwen3.6 文档（模型卡、验证、C8、W8A8 复现、WY）
docs/qwen38/     Qwen3.8 文档
docs/            共享文档（来源、代码地图）
results/qwen36/  Qwen3.6 原始数据与实验记录（索引见 results/README.md）
results/qwen38/  Qwen3.8 原始数据与实验记录
deploy/          三套部署档（W8A8+MTP / W8A8 / W4A8），见 deploy/README.md
tools/ scripts/ experimental/ native-overlay/ runtime/ bootstrap/ benchmarks/ patches/
```

结果目录名带日期，不改写；旧版本 commit 中的路径仍可通过 Git 历史访问。

## 复现

两条路线，按要跑哪一档选：

- **W8A8 TP2 / 262K（已发布镜像，不编译算子）**：[W8A8 262K 镜像拉取与逐步复现](docs/qwen36/w8a8-262k.md)。
- **W4A8 TP1 / 从源码构建**：[Qwen3.6 文档](docs/qwen36/README.md)。

两条路线都需要 Ascend 310P 硬件与驱动（不随仓库分发）。

## 已知限制

- 默认 WY/MTP2 主 Graph 使用已验收的 310P 兼容层；C8 等其他实验档位不属于默认交付。
- 公开 `build-native.sh` 未在全新环境逐字端到端验收（见 release-validation）。
- 性能数据按各记录的固定协议解释；本次 mask 增量精度门禁为同一真实算子输出对照，范围与结果随发布记录保存。
- 本仓库维护内部引擎交付源码、镜像版本和验收记录；站点配置与凭据由各部署环境维护。

License: Apache-2.0。第三方许可见 NOTICE。
