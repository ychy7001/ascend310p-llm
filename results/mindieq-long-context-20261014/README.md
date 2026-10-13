# MindIEQ 长上下文对照（2026-10-14）

两条服务均为 TP1、W4A8、动态 C8、GDN、MTP2、Decode Graph、prefix cache off；MindIEQ 使用 a2 chip0/18024，原始 W4 使用 chip1/18028，使用同一批 token，按顺序各执行一次。结果是单次长请求门禁，不是 P50/P95。

| prompt/output | MindIEQ Prefill | 原始 W4 Prefill | Prefill 提升 | MindIEQ Decode | 原始 W4 Decode |
|---:|---:|---:|---:|---:|---:|
| 64512/128 | 463.77 tok/s | 394.39 tok/s | **+17.59%** | 22.64 tok/s | 22.58 tok/s |
| 114560/14 | 377.18 tok/s | 328.94 tok/s | **+14.66%** | 17.33 tok/s | 17.35 tok/s |

64K 请求 TTFT 分别为 139.15s/163.69s；近 112K 请求 TTFT 分别为 303.79s/348.33s。两边均完整返回目标 token 数，无 OOM、KV 分配失败或 HTTP 错误。

## 解释

MindIEQ 的 Prefill 优势随上下文增长而收窄，但在 112K 产品规格仍保持约 14.7% 的单流 Prefill 收益；Decode 基本相同。因此可以把 MindIEQ 定位为长 Prefill 分流优化，默认阈值建议继续保留为 128 token 以上的实验开关，并在产品压测中分别报告 C8/原始 W4 的长 Prefill 与 Decode。

原始数据：`long64-pair.out`、`long112-pair.out`；服务日志：`mindieq-112k-service.log`、`original-112k-service.log`。
