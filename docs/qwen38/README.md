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

## 下载与许可

- 检查点 revision `d32978ccdde1729caec0034571b03a14779698e9`（约 169 GiB，含约 95 GiB 浮点 PLE）。国内可用 hf-mirror，[tools/qwen38_download_checkpoint.py](../../tools/qwen38_download_checkpoint.py) 默认 `--endpoint https://hf-mirror.com`，支持断点续传和分片并行。
- 本项目**不转载**该权重，避免与上游版本脱节，也避免在许可证上承担再分发责任。
- **许可证**：检查点与基础模型 Qwen/Qwen3.8-Flash-Next 均为 *Qwen Community License 1.0*（不是 Apache-2.0）。允许使用、修改和再分发（需保留版权声明）；但若使用者或其关联方经营 *Model as a Service* 或 *AI Work Assistant* 业务并用于商业目的，须另行向 Qwen 取得授权，仅内部使用且不向第三方开放除外。研究与个人复现不受影响；**用于对外提供推理服务的商业场景前须先核对该条款**。

## 复现限制

实验在 ychy-a2 上完成，原始日志已入库；a2 当前离线，未在其他主机重跑。权重与镜像不随仓库分发。
