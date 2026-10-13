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

该结果只证明当前服务在本次复测中运行正常。它不能单独构成 MindIEQ 相对 W4/W8A8 的因果加速结论，因为服务同时启用了 GDN、C8、MTP2 和 Graph；正式比较仍以 `results/mindieq-native-layout-20261013/` 的配对记录为准。
