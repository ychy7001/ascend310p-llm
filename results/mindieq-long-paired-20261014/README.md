# 64K MindIEQ / 原始 W4 单轮交替对照（2026-10-14）

这是 64K 长上下文的严格同 token 输入、同一输出长度的两轮交替样本。两个服务分别占用 a2 的两颗同型号 310P，均为 TP1、W4A8、C8、GDN、MTP2、Decode Graph、prefix cache off；请求使用非流式 completion，避免实验镜像在长流式响应结束时偶发不关闭连接。

| 服务 | Prefill | Decode | 总耗时 | usage |
|---|---:|---:|---:|---|
| MindIEQ | **483.80 tok/s 中位** | **22.64 tok/s 中位** | 138.97 s 中位 | 64512 + 128（2次） |
| 原始 W4 | **395.69 tok/s 中位** | **22.36 tok/s 中位** | 168.73 s 中位 | 64512 + 128（2次） |

两轮配对的 Prefill 中位提升为 **22.27%**，Decode 变化 **+1.25%**；4 个请求的输出 SHA256 完全一致。Prometheus 阶段计时、wall time、usage 和输出哈希保存在 `paired-64k-ns-v3.jsonl`。

这是 64K 的两轮中位数，不等同于正式线上 P50/P95；仍需更多轮次和并发条件。旧版 benchmark 曾在下一轮前被长连接/metrics 读取阻塞，修复连接处理后两轮均完整返回，容器日志最终回到 `Running: 0`。

## 32K 连续交替补充

重启 chip1 原始 W4 服务后，使用相同的非流式协议执行两轮交替（共 4 个正式请求）：

| 服务 | Prefill 中位 | Decode 中位 | 请求数 | 生命周期 |
|---|---:|---:|---:|---|
| MindIEQ | 569.11 tok/s | 29.12 tok/s | 2 | 两轮后 `Running: 0` |
| 原始 W4 | 453.21 tok/s | 28.79 tok/s | 2 | 两轮后 `Running: 0` |

相对原始 W4，32K Prefill 提升 **25.57%**，Decode 变化 **+1.15%**；四个请求 usage 均为 `32256+128`，输出哈希一致。原始数据见 `paired-32k-ns.jsonl`。

## 48K 单轮交替补充

修正 metrics 连接关闭逻辑后，执行一轮 48K 配对：MindIEQ `521.49/25.34 tok/s`，原始 W4 `420.77/24.93 tok/s`（Prefill/Decode），Prefill 提升 **23.94%**，Decode 变化 **+1.66%**。两边 usage 均为 `49152+128`、输出哈希一致，容器日志最终均为 `Running: 0`。原始数据见 `paired-48k-ns-v3.jsonl`。
