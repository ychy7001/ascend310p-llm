# MindIEQ 与原始 W4A8+C8 端到端对照（2026-10-14）

本目录保存 a2 两颗 310P 上的独立服务复测，用于验证 MindIEQ grouped Prefill 的实际增量。两组均为 TP1、W4A8、C8、GDN、MTP2、Decode Graph、prefix cache off、输出 128 token；MindIEQ 组使用 chip0/18024 的 E5 shared-gate 服务，原始组使用 chip1/18028 的独立服务并关闭 `QWEN36_MINDIEQ_ENABLED`。原始组在 8K 和 32K 各 warmup 1 次、正式测量 3 次，采集器在收到 usage 后结束流，避免 `[DONE]` 标记问题。

## 结果

| prompt | 原始 W4 Prefill | MindIEQ E5 Prefill | Prefill 增量 | 原始 W4 Decode | MindIEQ E5 Decode | Decode 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 525.88 tok/s | 624.77 tok/s | +18.81% | 33.73 tok/s | 34.68 tok/s | -2.83% |
| 32256 | 473.02 tok/s | 546.62 tok/s | +15.56% | 29.34 tok/s | 28.96 tok/s | -1.30% |

MindIEQ 数字来自 `results/qwen36/mindieq-shared-gate-20261014/service-comparison.json` 中的 E5 复测；原始 W4 数字为本目录三次中位数。两组不是时间交错的 A/B，可能包含服务启动时刻、温度和设备状态差异，因此该结果证明“值得继续做严格交替 A/B”，还不能作为最终收益承诺。

## 解释

这次复测比“MindIEQ 对上一版 grouped”的约 2.5% 增量更接近产品决策所关心的比较：原始 W4 与 MindIEQ 完整服务路径之间，长 Prefill 约有 15%～19% 的方向性提升。Decode 没有同方向提升，8K/32K 都有约 1%～3% 波动，说明 MindIEQ 应定位为 Prefill 专用优化，不能宣称端到端 Decode 加速。

下一步门禁是同一颗芯片停止/切换服务后交替执行 A/B，各 token 长度至少 5 次，并同时记录温度、NPU 利用率、TTFT、MTP 接受率和生成文本；如果 Prefill 仍稳定超过 10% 且 Decode/质量无回退，再考虑默认开启。

原始证据：`original-w4-c8-chip1.jsonl`、`original-w4-c8-service.log`。MindIEQ 对照证据：`../mindieq-shared-gate-20261014/`。
