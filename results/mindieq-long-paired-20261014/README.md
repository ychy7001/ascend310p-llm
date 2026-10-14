# 64K MindIEQ / 原始 W4 交替对照（2026-10-14）

这是 64K 长上下文的严格同 token 输入、同一输出长度的五轮交替样本。两个服务分别占用 a2 的两颗同型号 310P，均为 TP1、W4A8、C8、GDN、MTP2、Decode Graph、prefix cache off；请求使用非流式 completion，避免实验镜像在长流式响应结束时偶发不关闭连接。

| 服务 | Prefill | Decode | 总耗时 | usage |
|---|---:|---:|---:|---|
| MindIEQ | **483.35 tok/s 中位** | **22.60 tok/s 中位** | 139.10 s 中位 | 64512 + 128（5次） |
| 原始 W4 | **395.66 tok/s 中位** | **22.40 tok/s 中位** | 168.74 s 中位 | 64512 + 128（5次） |

五轮配对的 Prefill 中位提升为 **22.16%**，Decode 变化 **+0.92%**；10 个请求的输出 SHA256 完全一致。P95 分别为 MindIEQ `483.93/22.62 tok/s`、原始 W4 `395.72/22.42 tok/s`（Prefill/Decode）。Prometheus 阶段计时、wall time、usage 和输出哈希保存在 `paired-64k-ns-v4.jsonl`。

这是单流交替请求的 P50/P95，不等同于并发线上尾延迟。旧版 benchmark 曾在下一轮前被长连接/metrics 读取阻塞；本轮使用修正版客户端，五轮均完整返回，容器日志最终回到 `Running: 0`。

## 112K 复测说明

使用离线复用的同一批 `114560` 个 token ID，非流式 completion、输出上限 32、MindIEQ 与原始 W4 交替三轮。MindIEQ Prefill 中位为 **378.34 tok/s**，原始 W4 为 **329.53 tok/s**，提升 **14.81%**；Decode 中位分别为 **16.96/16.83 tok/s**，变化 **+0.75%**。总耗时中位为 **304.64/349.50 s**。六次 usage 均为 `114560+32`，输出 SHA256 完全一致，服务最终均回到 `Running: 0`。原始记录见 `paired-112k-fixed-v1.jsonl`。

可复现实验的确定性输入保存在 `prompt-114560.json.gz`；客户端 `tools/bench_fixed_tokens_paired.py` 同时支持普通 JSON 和 gzip JSON。示例：

```bash
python3 tools/bench_fixed_tokens_paired.py \
  --mindieq-url http://127.0.0.1:18024 \
  --original-url http://127.0.0.1:18028 \
  --model qwen36-w4a8-c8 \
  --prompt-file results/mindieq-long-paired-20261014/prompt-114560.json.gz \
  --out /tmp/paired-112k.jsonl --output-tokens 32 --repeats 3
```

这组结果是三轮单流阶段吞吐验证，不是线上并发 P95，也不包含流式 TTFT；Prometheus phase delta、wall time 和输出哈希均保留在 JSONL 中。早期客户端在 token 准备/响应期间的缓冲文件和服务访问日志不足以证明 tokenizer 阻塞，不能再据此作故障归因。

## 32K 连续交替补充

重启 chip1 原始 W4 服务后，使用相同的非流式协议执行两轮交替（共 4 个正式请求）：

| 服务 | Prefill 中位 | Decode 中位 | 请求数 | 生命周期 |
|---|---:|---:|---:|---|
| MindIEQ | 569.11 tok/s | 29.12 tok/s | 2 | 两轮后 `Running: 0` |
| 原始 W4 | 453.21 tok/s | 28.79 tok/s | 2 | 两轮后 `Running: 0` |

相对原始 W4，32K Prefill 提升 **25.57%**，Decode 变化 **+1.15%**；四个请求 usage 均为 `32256+128`，输出哈希一致。原始数据见 `paired-32k-ns.jsonl`。

## 48K 单轮交替补充

修正 metrics 连接关闭逻辑后，执行一轮 48K 配对：MindIEQ `521.49/25.34 tok/s`，原始 W4 `420.77/24.93 tok/s`（Prefill/Decode），Prefill 提升 **23.94%**，Decode 变化 **+1.66%**。两边 usage 均为 `49152+128`、输出哈希一致，容器日志最终均为 `Running: 0`。原始数据见 `paired-48k-ns-v3.jsonl`。
