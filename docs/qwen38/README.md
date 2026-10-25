# Qwen3.8-Flash-Next on dual Ascend 310P（TP2/EP2）

在一张 Atlas 300I DUO（两颗 310P3）上运行 Qwen3.8-Flash-Next：TP2/EP2，routed experts 为原生 G128 W4A8，其余层 FP16，C16，纯文本，MTP0。社区模型卡的实测是四颗 310P（TP4/EP4），本项目缩到两颗，依赖 PLE 按需 mmap 读取。当前是实验档，实测到 16K。

## 当前状态

| 项目 | 状态 | 记录 |
| --- | --- | --- |
| 容量审计 | 完成：专家 61.52 GiB，TP1 装不进单颗 48G，TP2 每 rank 权重约 37 GiB | [feasibility](../../results/qwen38/tp2-feasibility-20261018/README.md) |
| 真实专家 / 融合 MLP / TP2 QSA / PLE lazy | 通过 | [validation](../../results/qwen38/tp2-validation-20261018/README.md) |
| 完整检查点 TP2 服务，4K | 通过，Decode 13.82–13.89 tok/s（512/128） | [serving](../../results/qwen38/tp2-serving-20261018/README.md) |
| Decode Graph | 通过，512/128 约 +20% | [decode-graph](../../results/qwen38/decode-graph-20261018/README.md) |
| 16K（16256+128） | 通过，Decode 15.59 tok/s，TTFT 约 69 s | [long-context](../../results/qwen38/long-context-20261018/README.md) |
| 32K | 可启动，但 32256+128 请求无法调度 | [32k](../../results/qwen38/long-context-32k-20261018/README.md) |
| batch512 预填充 | 8K 更快，16K 请求无法调度，不采用 | [batch512](../../results/qwen38/prefill-batch512-20261018/README.md) |
| MTP / C8 / Prefix Cache / 并发 / 完整质量评估 / 多模态 | 未验证 | — |

推荐配置：16K / batch256 / C16 / MTP0 / Decode Graph。

## 文档

- [跑通过程](bring-up.md)：从容量核算、算子验证、整模排障到 Decode Graph 和长上下文边界，按决策顺序讲。
- [GDN Prefill 因果掩码错误](gdn-causal-mask-bug.md)：适配中发现并修复的社区算子 bug。

## 来源

- 检查点：社区 matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i，固定 revision 见 serving 记录。
- 代码：opensensor/vllm-ascend `ffa980c1` + opensensor/vllm `3ab5dda29`，加本仓库三个补丁：
  - [gdn-causal-mask-vector.patch](../../patches/qwen38/gdn-causal-mask-vector.patch)：修复 Prefill 因果掩码
  - [compact-single-request-budget.patch](../../patches/qwen38/compact-single-request-budget.patch)：单请求预算
  - [hccl-buffer-experiment.patch](../../patches/qwen38/hccl-buffer-experiment.patch)：通信缓冲实验

## 下载与许可

- 检查点 revision `d32978ccdde1729caec0034571b03a14779698e9`（约 169 GiB，含约 95 GiB 浮点 PLE）。国内可用 hf-mirror，[tools/qwen38_download_checkpoint.py](../../tools/qwen38_download_checkpoint.py) 默认 `--endpoint https://hf-mirror.com`，支持断点续传和分片并行。
- 本项目不转载该权重，避免与上游版本脱节，也避免在许可证上承担再分发责任。
- **许可证**：检查点与基础模型 Qwen/Qwen3.8-Flash-Next 均为 *Qwen Community License 1.0*（不是 Apache-2.0）。允许使用、修改和再分发（需保留版权声明）；但若使用者或其关联方经营 *Model as a Service* 或 *AI Work Assistant* 业务并用于商业目的，须另行向 Qwen 取得授权，仅内部使用且不向第三方开放除外。研究与个人复现不受影响；用于对外提供推理服务的商业场景前须先核对该条款。

## 复现范围

实验在 ychy-a2 上完成，原始日志已入库；a2 当前离线，未在其他主机重跑。命令使用 rootless Podman，Docker 用户需自行调整设备挂载和用户映射，仓库未在 Docker 上验证过启动流程。权重与镜像不随仓库分发。
