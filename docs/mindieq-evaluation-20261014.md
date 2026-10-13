# MindIEQ grouped Prefill 评估与投入决策

## 当前证据

MindIEQ 的实现位于 `experimental/mindieq_grouped_kernel/` 和 `experimental/mindieq_prefill_adapter.py`，只替换 TP1 routed-expert 的长 Prefill grouped 路径；Decode、Down projection 和 C8 KV 路径不变。共享 Gate 版本额外复用 Gate/Up 的 A8 激活和 scale，保留原 W4 权重存储。

需要区分两个比较基线：

1. 早期原始 W4 路径到后续 grouped 路径的历史提升约 22%～25%，说明 grouped 调度和布局本身值得研究。
2. 在已经启用 grouped 路径的前一版本上，2026-10-14 的 E5 shared-gate 复测才是 MindIEQ 当前增量收益：

| 冷前缀 | 前一版本 Prefill | E5 Prefill | 增量 | Decode 增量 | 文本一致数 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 607.46 tok/s | 624.77 tok/s | +2.85% | -0.40% | 3/3 |
| 32256 | 533.06 tok/s | 546.62 tok/s | +2.54% | +0.03% | 3/3 |

测试为独立服务、prefix off、warmup 1、repeat 3，并非交替 A/B；所以应把上述结果视为方向性增益，而不是正式置信区间。

质量门禁中，固定 64 题子集两边均为 63/64，`answer_changes=[]`；这是小样本等价证据，不是全面精度证明。共享 Gate 的 17/32/128/512 token 设备微测中，A8 codes/scale 和 Gate output 均 bitwise identical；512 token 的 Gate+准备中位耗时由 4.406ms 降为 3.172ms，说明局部准备开销有实际收益。

完整原始数据：`results/mindieq-shared-gate-20261014/service-comparison.json`、`shared-gate-v1-20261014.txt`。

## 决策

MindIEQ **值得保留并进入产品候选评估**。当前最可靠的回退基线仍是原 W4A8 + C8 + GDN + MTP2 + Decode Graph；MindIEQ 作为 `MIN_TOKENS=128` 的长 Prefill 分流，短输入回退原 W4。由于 P95、完整质量集、TP2/C8 和并发门禁尚未完成，暂不把它写成无条件默认后端。

继续投入的理由是：

- 真实模型服务上没有观察到有意义的 Decode 回退；
- 相对原始 W4 的交替五次复测，8K/32K Prefill 中位提升 20.69%～23.29%，64K/近112K单次提升 17.59%/14.66%；相对前一 grouped 版本的 E5 shared-gate 增量约 2.5%～2.9%只是增量优化，不能替代原始 W4 对照；
- 量化共享路径已有 bitwise parity 和固定题集门禁；
- 后续可以把收益转化为更少的 launch、一次性 expert dispatch 或更好的大 token 数布局。

暂不默认开启的理由是：

- 当前端到端收益小于测量和服务间波动，尚未完成交替 A/B；
- 依赖外部 MindIEQ 制品和实验性 monkeypatch；
- 只验证 TP1、FP16 激活、固定 checkpoint 形状，尚不支持 TP2/C8 多请求；
- 64 题质量集和有限业务样本不足以覆盖长上下文、多任务和异常路由。

## 下一道门

只有同时满足以下条件，才考虑将 MindIEQ 作为长 Prefill 默认路径或单独提交上游：

- 同一芯片或可控隔离条件下交替 A/B，8K/32K/64K/112K 各至少 5 次，Prefill P50 稳定提升 ≥10%；
- Decode tok/s、TTFT、MTP 接受率无统计显著退化；
- 公开质量集合和长上下文 needle 均通过，且无输出漂移；
- 明确声明 TP1、单流、C8 约束，并补齐制品 hash、许可证和回退路径。

在达到门槛前，优先做 kernel/dispatch profile 和交替 A/B，不再盲目扩展完整模型集成范围。

## 交替五次复测（新增）

同一批 token 在 a2 两颗同型号 310P 的 `18024`（MindIEQ E5）和 `18028`（原始 W4）之间轮换，各 prompt 长度五次。8K 的 Prefill 中位为 620.27 vs 503.09 tok/s（+23.29%），32K 为 541.99 vs 449.07 tok/s（+20.69%）；Decode 变化分别为 -0.06% 和 -0.48%。这比独立服务的前一轮结果更稳定，足以把 MindIEQ 列为长 Prefill 产品候选，但仍需 64K/112K、P95 和完整质量门禁。

## 长上下文门禁（新增）

同配置单流长请求显示收益随上下文增长而收窄但仍为正：64K（64512/128）MindIEQ 463.77 vs 原始 W4 394.39 tok/s，+17.59%；近 112K（114560/14）为 377.18 vs 328.94 tok/s，+14.66%。两边 Decode 分别约 22.6 和 17.3 tok/s，差异小于 0.1%，请求均完整返回。该结果支持在 112K 产品规格中保留 MindIEQ 长 Prefill 分流，但仍不覆盖并发和尾延迟。

随后使用非流式 completion 做了一组严格同 token 的 64K 交替补充：MindIEQ `483.46 tok/s`、原始 W4 `395.70 tok/s`，Prefill `+22.18%`；Decode `22.61` vs `22.68 tok/s`，输出哈希一致。当前实验服务在下一轮长请求上出现调度停滞，因此这组数据只作为单轮因果补充，不能替代 64K 的多次 P50/P95 门禁。详见 `results/mindieq-long-paired-20261014/`。

## 112K needle 正确性（新增）

在同样的 114560 token 文档、needle 位于 73% 位置、输出上限 14 的请求中，MindIEQ 和原始 W4 均返回 `ASCEND732941`，usage 和 stop 原因一致。MindIEQ 请求耗时 298.08s，原始 W4 348.15s。该 canary 支持长上下文路径没有明显检索破坏，但不替代完整质量集和多位置 needle 测试。

## 32K 多位置 needle（新增）

在 25%、50%、75%、90% 四个位置各放置同一检索码，MindIEQ 和原始 W4 共 8 次请求全部返回正确结果，usage 和 stop 原因一致。该门禁减少了单一 needle 位置的偶然性，但仍不是完整任务质量集。
