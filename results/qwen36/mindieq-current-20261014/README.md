# 当前 MindIEQ native-layout 服务复测（2026-10-14）

## 配置

- a2 chip0 / 单 310P3
- TP1、48G、W4A8、动态 C8、MTP2、Decode Graph
- GDN PR #17640 固定算子开启
- MindIEQ native-layout grouped 后端开启
- `max_model_len=114688`、`gpu_memory_utilization=0.95`
- `max_num_batched_tokens=4096`、`max_num_seqs=1`、prefix cache 关闭
- 服务端口 `18024`，独占服务；warmup 1，正式请求 3 次

## 8K/128 结果

使用 `tools/bench_server_phases.py`，同一组 8192 token IDs，Prometheus 请求阶段计时：

| 指标 | 三次结果 | 中位数 |
|---|---:|---:|
| Prefill tok/s | 625.267 / 625.386 / 625.149 | **625.267** |
| Decode tok/s | 35.811 / 35.867 / 35.918 | **35.867** |
| TTFT | 13.11 s 左右 | — |

每次均通过计数器和 usage 校验。MTP 计数完全一致：draft 44 轮、draft token 88、accepted token 84，接受率 95.45%，平均接受长度 2.909。原始记录见 `current-8k-20261014.jsonl`。

该结果只证明当前服务在本次复测中运行正常。它不能单独构成 MindIEQ 相对 W4/W8A8 的因果加速结论，因为服务同时启用了 GDN、C8、MTP2 和 Graph；正式比较仍以 `results/qwen36/mindieq-native-layout-20261013` 的配对记录为准。

## 32K/128 结果

同一服务继续执行 32256 token、输出 128、warmup 1 + 正式 3 次：

| 指标 | 三次结果 | 中位数 |
|---|---:|---:|
| Prefill tok/s | 546.851 / 546.397 / 545.844 | **546.397** |
| Decode tok/s | 29.185 / 29.153 / 29.136 | **29.153** |

三次请求均通过计数器/usage 校验；MTP 每次 43 drafts、86 drafted、86 accepted，接受率 100%，平均接受长度 3。相对本次 8K 复测，32K Prefill 下降约 12.6%，Decode 下降约 18.7%；这反映长 KV/注意力成本，不能归因于 MindIEQ 单独开销。

## 114560 token 近满长检索

当前服务在 `max_model_len=114688` 下执行 114560 token、输出 14、warmup 1 + 正式 1 次，Prefill **378.373 tok/s**，Decode **17.339 tok/s**，Prefill 阶段耗时 302.770 s，Decode 阶段耗时 0.750 s。MTP 为 5 drafts、10 drafted、10 accepted，接受率 100%，平均接受长度 3；计数器和 usage 校验通过。该请求用于容量和阶段性能确认，未注入专门 needle，因此不能作为检索正确性证明；历史同配置 needle 正确性记录仍见 `results/qwen36/pr17640-b4096-20261013` 和 `results/qwen36/mindieq-native-layout-20261013`。

## 114560 token needle 正确性

使用 `tools/check_long_context_needle.py` 在约一半位置插入唯一验证码 `ASCEND732941`，114560 token 输入、最多 128 输出。实际耗时 **297.733 s**，返回 `ASCEND732941`，`retrieved=true`，usage 为 114560 prompt / 14 completion，正常 stop。该结果证明当前组合在近满上下文下可完成一次基本内容检索；它不是全面长上下文质量评估。
