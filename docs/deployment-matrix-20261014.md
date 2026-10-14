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
| MindIEQ | native-layout grouped，可选 | 112K 单流长 Prefill 候选；阈值128分流，短输入回退原W4，正式默认仍待质量/P95门禁 |
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

### MindIEQ 与原始 W4 的最新交替对照

为避免把早期“grouped 相对旧版本约 2.5%～2.9%”误读为最终收益，当前采用两颗同型号 310P 的交替五次复测作为主证据。两端使用相同 token 输入、prefix off、同一 112K 配置；该方法仍不是同一芯片同时运行的严格统计实验，但比单独分批测量更能抑制时间段波动。

| 输入 | MindIEQ Prefill 中位 | 原始 W4 Prefill 中位 | Prefill变化 | Decode变化 | 质量门禁 |
|---:|---:|---:|---:|---:|---|
| 8K/128 | 620.27 | 503.09 | **+23.29%** | -0.06% | 5/5文本一致 |
| 32K/128 | 541.99 | 449.07 | **+20.69%** | -0.48% | 5/5文本一致 |
| 64K/128 | 463.77 | 394.39 | **+17.59%** | 约0% | 两边完整返回 |
| 112K近上限 | 377.18 | 328.94 | **+14.66%** | 约0% | needle均正确 |

严格同 token 的 64K 非流式两轮配对为：MindIEQ `483.80/22.64 tok/s` 中位，原始 W4 `395.69/22.36 tok/s` 中位，Prefill `+22.27%`，Decode `+1.25%`，4 个请求输出哈希一致。原始数据见 `results/mindieq-long-paired-20261014/`；这是单流两轮中位数，不是线上 P50/P95。

修复 metrics 连接处理后，48K 单轮配对为 MindIEQ `521.49/25.34 tok/s`、原始 W4 `420.77/24.93 tok/s`，Prefill `+23.94%`，两边最终均回到 `Running: 0`。

因此 MindIEQ 已经值得保留为长 Prefill 产品候选，但不能据此宣称 W4 全面加速：它只替换 routed-expert 的 Prefill grouped 路径，Decode、C8 attention 和并发路径仍使用原实现。当前部署策略是 `QWEN36_MINDIEQ_MIN_TOKENS=128`，短输入回退原W4；正式默认开启前还需要 P95、完整质量集、TP2/C8 和多请求门禁。

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

FP16 KV 并发备选可以复用同一入口，将 C8 关闭并显式指定设备和并发数；默认脚本值仍为 chip0、单请求和 C8：

```bash
QWEN36_C8_ENABLED=0 MAX_NUM_SEQS=8 NPU_DEVICE=/dev/davinci1 \
MAX_MODEL_LEN=32768 MEMORY_UTILIZATION=0.90 ENFORCE_EAGER=1 \
...同上启动参数... bash experimental/run-c8-112k-product.sh
```

这组参数只打开可复现入口，不代表 FP16 KV 已支持 112K 或 Graph 并发；当前实测门禁是 32K 和短请求并发。

## 取舍和未完成门禁

- MindIEQ native-layout grouped 在真实 256 expert MLP 和完整服务中显示出 Prefill 潜力；交替五次复测相对原始 W4 在 8K/32K 提升 20.69%～23.29%，64K/近112K单次长请求提升 17.59%/14.66%。这些是单流阶段吞吐结果，不是并发或 P95 承诺。
- Decode 的主要成本仍是长 KV/Attention、C8 转换和调度；MindIEQ 没有证明稳定的 Decode 提升。
- C8、Graph 和 GDN 均依赖实验性外部制品或 monkeypatch；默认启动器保留开关，不能称为官方 backend 支持。
- 旧版 64K 配对中出现过客户端长连接/metrics 读取阻塞；容器日志随后回到 `Running: 0`，不能据此判定服务端泄漏。已修复 phase benchmark 的连接处理，64K 两轮配对均完整返回。
- 32K 两轮连续交替、48K 单轮交替和 64K 两轮交替均完成并回到 `Running: 0`；仍需更多轮次、P95 和并发门禁。
- 尚未完成：同条件 W8A8 交替 A/B、TP2 C8、32G 实机、并发 2/4/8 的完整矩阵、P50/P95、正式质量集合及多模态；MindIEQ 目前仍保持可选分流。

## 并发边界

2026-10-14 在 a2 chip1 用隔离服务尝试 `max_num_seqs=8`，并发2即触发 C8 adapter 的 `C8 requires real four-tensor storage and one request`，EngineCore退出。当前 C8 后端因此明确只支持单请求；不能将单流吞吐外推为并发2/4/8。完整失败证据见 `results/concurrency-c8-limit-20261014/`。并发支持需要重新设计按 sequence 分离的四张 cache storage 和 batch attention 路径。

作为替代路线，关闭 C8 后的 W4A8+FP16 KV 在 chip1 的 eager/max_num_seqs=8 服务上并发2/4/8均正常：短512/32请求总吞吐中位分别为19.873/23.790/25.227 tok/s，但TTFT升至2.13/4.10/8.08s，且该路径放弃C8的KV节省与112K单流容量。原始数据见 `results/fp16kv-concurrency-20261014/`，只能作为并发备选探针。

FP16 KV 并发备选的单流 32K/128 复测为 Prefill **453.303 tok/s**、Decode **29.126 tok/s**，MTP 接受率100%；服务 `max_model_len=32768`，因此该结果支持“约32K并发备选”，不支持更长上下文外推。

### FP16 KV 64K 单流边界（新增）

为验证并发备选是否只能停留在 32K，a2 chip1 启动了独立 `18027` 服务：W4A8、FP16 KV（C8 关闭）、MTP2、GDN、eager、`max_num_seqs=1`、`max_model_len=65536`。服务健康检查通过，启动 profiling 报告模型权重 27.29GiB、可用 KV budget 3.20GiB、GPU KV cache size 131072 tokens。

对冷前缀 `64512/128` 做一次修正后的单次采集（服务端已返回 usage，绕开流式 `[DONE]` 结束标记问题）得到：TTFT 162.696s，prefill phase 162.670s，即 **396.4 tok/s**；decode phase 5.118s/127 tokens，即 **24.81 tok/s**。请求完整返回 64512 prompt + 128 completion，无 OOM 和错误。原始记录、服务日志和采集器输出在 `results/fp16kv-concurrency-20261014/`。

因此当前规格判断是：C8 单流仍采用 112K 产品基线；FP16 KV 可作为保留并发/兼容路线，单流 64K 已有实证，但其长 prefill（约 396 tok/s）不适合直接替代 C8 的长上下文产品路径。64K 结果也不能外推到并发 2/4/8。
