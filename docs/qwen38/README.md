# Qwen3.8-Flash-Next on dual Ascend 310P（TP2/EP2）

状态：**实验，仅文本、MTP0**。验证范围以各结果目录为准，不沿用 Qwen3.6 的规格。

| 项目 | 状态 | 记录 |
| --- | --- | --- |
| 容量审计与可行性 | 完成 | [feasibility](../../results/qwen38/tp2-feasibility-20261018/README.md) |
| 真实专家 / 融合 MLP / TP2 QSA / PLE lazy | 通过 | [validation](../../results/qwen38/tp2-validation-20261018/README.md) |
| 完整检查点 TP2 服务，4K 请求 | 通过 | [serving](../../results/qwen38/tp2-serving-20261018/README.md) |
| Decode Graph（512/128 约 +20%） | 通过 | [decode-graph](../../results/qwen38/decode-graph-20261018/README.md) |
| 16K（16256+128） | 通过 | [long-context](../../results/qwen38/long-context-20261018/README.md) |
| 32K | 可启动但 32256+128 无法调度，**不可用** | [32k](../../results/qwen38/long-context-32k-20261018/README.md) |
| batch512 预填充 | 已对照 | [batch512](../../results/qwen38/prefill-batch512-20261018/README.md) |
| MTP / C8 / 并发 / 完整质量 / 多模态 | **未验证** | — |

## 来源

- 检查点：社区 matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i（固定 revision 见 serving 记录）。
- 代码：opensensor/vllm-ascend `ffa980c1` + opensensor/vllm `3ab5dda29`，加本仓库三个补丁：
  - [gdn-causal-mask-vector.patch](../../patches/qwen38/gdn-causal-mask-vector.patch)：修复 Prefill 因果掩码
  - [compact-single-request-budget.patch](../../patches/qwen38/compact-single-request-budget.patch)：单请求预算
  - [hccl-buffer-experiment.patch](../../patches/qwen38/hccl-buffer-experiment.patch)：通信缓冲实验
- 社区模型卡实测为四颗 310P（TP4/EP4）；本项目缩到两颗，依赖 PLE lazy 读取，冷读性能见各记录。

## 复现限制

实验在 ychy-a2 上完成，原始日志已入库；a2 当前离线，未在其他主机重跑。权重与镜像不随仓库分发。
