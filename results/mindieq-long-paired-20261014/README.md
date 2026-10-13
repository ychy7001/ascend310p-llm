# 64K MindIEQ / 原始 W4 单轮交替对照（2026-10-14）

这是 64K 长上下文的一组严格同 token 输入、同一输出长度的交替样本。两个服务分别占用 a2 的两颗同型号 310P，均为 TP1、W4A8、C8、GDN、MTP2、Decode Graph、prefix cache off；请求使用非流式 completion，避免实验镜像在长流式响应结束时偶发不关闭连接。

| 服务 | Prefill | Decode | 总耗时 | usage |
|---|---:|---:|---:|---|
| MindIEQ | 483.46 tok/s | 22.61 tok/s | 139.06 s | 64512 + 128 |
| 原始 W4 | 395.70 tok/s | 22.68 tok/s | 168.64 s | 64512 + 128 |

该单轮配对的 Prefill 提升为 **22.18%**，Decode 变化约 **-0.30%**；两边输出 SHA256 一致。Prometheus 阶段计时、wall time、usage 和输出哈希保存在 `paired-64k-ns-partial.jsonl`。

这是补充证据，不是 64K 的 P50/P95：后续重复请求在当前实验服务上出现长请求调度停滞，因此没有把未完成的轮次写入汇总。正式决策仍使用已有 8K/32K 五次交替、64K/112K 单次长请求和质量门禁。
