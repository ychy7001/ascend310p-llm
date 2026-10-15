# orangepi RC 基线快照（2026-10-15）

本目录记录 `orangepi` SSH 别名对应的 RC 实验端点。该端点与 `ychy-a2` 的 EP 实验分开统计。

## 端点

- 主机别名：`orangepi`
- 可见 NPU：单颗 `310P1`
- 容器：`qwen36-orangepi`
- 服务：`http://127.0.0.1:18024`
- 健康检查：2026-10-15 00:36 UTC 返回 HTTP 200

## 当前环境

```text
VLLM_ASCEND_FORCE_RC=1
QWEN36_C8_ENABLED=1
QWEN36_C8_DEVICE_LENGTH=1
QWEN36_C8_FUSED_WRITE=1
QWEN36_C8_FORCE_UPDATABLE_GRAPH=1
QWEN36_C8_POOL_MODE=split_by_type
QWEN36_MINDIEQ_ENABLED=1
QWEN36_MINDIEQ_CUSTOM_QUANT=1
QWEN36_MINDIEQ_MIN_TOKENS=128
GRAPH=1
MTP=1
GDN=1
ASYNC_SCHEDULING=0
MAX_MODEL_LEN=114688
KV_BYTES=6442450944
MAX_BATCHED_TOKENS=2048
```

启动脚本通过 `FULL_DECODE_ONLY` Graph 模式捕获 MTP1 的 batch `[1,2]`，服务使用 TP1、W4A8、自定义 Routed Expert 算子、C8 KV、Prefix Cache 和 MindIEQ 候选路径。

## 既有 RC 参考值

这些是单流、无 Prefix Cache 命中的参考值，后续 stage1 需要在同一端点重复确认：

| 输入长度 | Prefill | Decode | 备注 |
| ---: | ---: | ---: | --- |
| 8192 | 约 565 tok/s | 约 24.0 tok/s | MTP1、C8、Graph |
| 32256 | 约 511 tok/s | 约 19.3 tok/s | MTP1、C8、Graph |
| 114560 | 约 400 tok/s | 约 11.2 tok/s | 单次 near-maximum gate |

## 统计约束

- RC 结果不与 A2 EP 结果合并计算。
- 每组测试固定 warmup、使用独立 prompt，并保留原始 JSONL、环境快照和容器日志。
- 当前只把 112K 作为产品候选上下文；114560 需要补充重复测试后才能作为稳定性结论。
- 当前 C8 适配器仍按单请求路径验证，未宣称并发 2/4/8。

