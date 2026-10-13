# Qwen3.6 310P 部署矩阵（2026-10-14）

本文把当前可复现配置和实验开关分开记录。所有速度都是 a2 单颗 310P3、TP1、单流、warmup 后的独占端点结果；不能外推到 TP2、32G、多请求并发或 W8A8。

## 推荐基线

| 项目 | 值 | 状态 |
|---|---|---|
| 权重 | Qwen3.6-35B-A3B W4A8，routed experts INT4，MTP 保留 BF16 | 已加载验证 |
| 芯片 | 1× Ascend 310P3 / 48G | 已验证 |
| KV | 动态 C8/INT8 | 实验后端，已做 8K/32K/114560 门禁 |
| MTP | 2 draft tokens | 当前单流候选；真实文本接受率仍需更广采样 |
| Graph | FULL_DECODE_ONLY，capture 1/2/3，mode 0 | 实验性 updatable-graph 路径 |
| GDN | PR #17640 固定版本 | 310P GDN 优化候选 |
| MindIEQ | native-layout grouped，可选 | 推荐作为 Prefill 实验开关，默认仍关闭 |
| max_model_len | 114688 | 112K 产品规格建议；114560 仅近上限门禁 |
| gpu_memory_utilization | 0.95 | 48G TP1 已验证 |
| max_num_batched_tokens | 4096 | GDN 候选已验证；与 512 expert chunk 配套 |
| max_num_seqs | 1 | 只承诺单流 |
| prefix cache | 关闭 | 冷 Prefill 基线；可单独开启测试 |

## 当前组合的实测数据

当前服务同时开启 GDN、MindIEQ、C8、MTP2、Graph，因此下表是组合结果，不是某一个算子的独立收益：

| 输入/输出 | Prefill | Decode | MTP |
|---:|---:|---:|---|
| 8192/128 | 625.267 tok/s | 35.867 tok/s | 95.45%，平均接受长度2.909 |
| 32256/128 | 546.397 tok/s | 29.153 tok/s | 100%，平均接受长度3 |
| 114560/14 | 378.373 tok/s | 17.339 tok/s | 100%，平均接受长度3 |
| 114560 needle/14 | — | — | `ASCEND732941` 正确返回，297.733s |

114560 token 只证明一次近上限请求和基本检索可用；产品规格使用 112K 是为了给输出、调度和运行时波动留余量。

## 启动方式

沿用 `scripts/run-candidate.sh`，并显式提供固定制品。基础 W4A8+C8+GDN+MTP2：

```bash
MODEL_DIR=/abs/model \
OPP_DIR=/abs/w4-opp \
EXTENSION=/abs/qwen36_int4_custom.so \
C8_LIBRARY=/abs/libqwen36_c8.so \
GDN_OPP_DIR=/abs/pr17640-opp \
GDN_LIBRARY=/abs/pr17640_gdn.so \
GDN_SOURCE=/abs/chunk_gated_delta_rule.py \
PORT=18024 bash scripts/run-candidate.sh
```

在上述命令上增加 MindIEQ native-layout grouped（只建议用于 Prefill 对照）：

```bash
MINDIEQ_LIBRARY=/abs/mindieq_grouped.so \
QWEN36_MINDIEQ_ENABLED=1 \
QWEN36_MINDIEQ_CUSTOM_QUANT=1 \
QWEN36_MINDIEQ_SHARED_GATE=1 \
QWEN36_MINDIEQ_MIN_TOKENS=128 \
...同上启动参数... bash scripts/run-candidate.sh
```

`QWEN36_MINDIEQ_MIN_TOKENS=128` 是根据短输入开销选择的阈值；短请求走原 W4，长 Prefill 才走 grouped 路径。MindIEQ 当前仍需与原 W4 做 A/B 和更广文本质量比较，不能无条件打开。

## 取舍和未完成门禁

- MindIEQ native-layout grouped 在真实 256 expert MLP 和完整服务中显示出 Prefill 潜力，8K/32K 历史配对约有 22%～25% 提升；当前复测组合不提供独立因果证明。
- Decode 的主要成本仍是长 KV/Attention、C8 转换和调度；MindIEQ 没有证明稳定的 Decode 提升。
- C8、Graph 和 GDN 均依赖实验性外部制品或 monkeypatch；默认启动器保留开关，不能称为官方 backend 支持。
- 尚未完成：同条件 W8A8 交替 A/B、TP2 C8、32G 实机、并发 2/4/8 的完整矩阵、P50/P95、正式质量集合及多模态。

## 并发边界

2026-10-14 在 a2 chip1 用隔离服务尝试 `max_num_seqs=8`，并发2即触发 C8 adapter 的 `C8 requires real four-tensor storage and one request`，EngineCore退出。当前 C8 后端因此明确只支持单请求；不能将单流吞吐外推为并发2/4/8。完整失败证据见 `results/concurrency-c8-limit-20261014/`。并发支持需要重新设计按 sequence 分离的四张 cache storage 和 batch attention 路径。

作为替代路线，关闭 C8 后的 W4A8+FP16 KV 在 chip1 的 eager/max_num_seqs=8 服务上并发2/4/8均正常：短512/32请求总吞吐中位分别为19.873/23.790/25.227 tok/s，但TTFT升至2.13/4.10/8.08s，且该路径放弃C8的KV节省与112K单流容量。原始数据见 `results/fp16kv-concurrency-20261014/`，只能作为并发备选探针。
