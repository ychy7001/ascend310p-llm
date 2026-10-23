# 64K MindIEQ / 原始 W4 交替对照（2026-10-14）

这是 64K 长上下文的严格同 token 输入、同一输出长度的五轮交替样本。两个服务分别占用 a2 的两颗同型号 310P，均为 TP1、W4A8、C8、GDN、MTP2、Decode Graph、prefix cache off；请求使用非流式 completion，避免实验镜像在长流式响应结束时偶发不关闭连接。

| 服务 | Prefill | Decode | 总耗时 | usage |
|---|---:|---:|---:|---|
| MindIEQ | **483.35 tok/s 中位** | **22.60 tok/s 中位** | 139.10 s 中位 | 64512 + 128（5次） |
| 原始 W4 | **395.66 tok/s 中位** | **22.40 tok/s 中位** | 168.74 s 中位 | 64512 + 128（5次） |

五轮配对的 Prefill 中位提升为 **22.16%**，Decode 变化 **+0.92%**；10 个请求的输出 SHA256 完全一致。五个样本按 nearest-rank 得到的吞吐 P95 分别为 MindIEQ `483.93/22.62 tok/s`、原始 W4 `395.72/22.42 tok/s`；这是快端吞吐值，不能用于证明慢端尾延迟。wall-time 经验 P95 分别为 `139.13/168.78 s`，样本量不足以支撑线上 P95 承诺。Prometheus 阶段计时、wall time、usage 和输出哈希保存在 `paired-64k-ns-v4.jsonl`。

这是单流交替请求的 P50/P95，不等同于并发线上尾延迟。旧版 benchmark 曾在下一轮前被长连接/metrics 读取阻塞；本轮使用修正版客户端，五轮均完整返回，容器日志最终回到 `Running: 0`。

## 112K 复测说明

使用离线复用的同一批 `114560` 个 token ID，非流式 completion、输出上限 32、MindIEQ 与原始 W4 交替三轮。MindIEQ Prefill 中位为 **378.34 tok/s**，原始 W4 为 **329.53 tok/s**，提升 **14.81%**；Decode 中位分别为 **16.96/16.83 tok/s**，变化 **+0.75%**。总耗时中位为 **304.64/349.50 s**。六次 usage 均为 `114560+32`，输出 SHA256 完全一致，服务最终均回到 `Running: 0`。原始记录见 `paired-112k-fixed-v1.jsonl`。

可复现实验的确定性输入保存在 `prompt-114560.json.gz`；客户端 `tools/bench_fixed_tokens_paired.py` 同时支持普通 JSON 和 gzip JSON。示例：

```bash
python3 tools/bench_fixed_tokens_paired.py \
  --mindieq-url http://127.0.0.1:18024 \
  --original-url http://127.0.0.1:18028 \
  --model qwen36-w4a8-c8 \
  --prompt-file results/qwen36/mindieq-long-paired-20261014/prompt-114560.json.gz \
  --out /tmp/paired-112k.jsonl --output-tokens 32 --repeats 3
```

这组结果是三轮单流阶段吞吐验证，不是线上并发 P95，也不包含流式 TTFT；本轮保存了阶段吞吐、wall time、usage 和输出哈希；32-token 历史客户端未保存原始 phase delta，后续 128-token 版本已补齐。早期客户端在 token 准备/响应期间的缓冲文件和服务访问日志不足以证明 tokenizer 阻塞，不能再据此作故障归因。

## 32K 连续交替补充

重启 chip1 原始 W4 服务后，使用相同的非流式协议执行两轮交替（共 4 个正式请求）：

| 服务 | Prefill 中位 | Decode 中位 | 请求数 | 生命周期 |
|---|---:|---:|---:|---|
| MindIEQ | 569.11 tok/s | 29.12 tok/s | 2 | 两轮后 `Running: 0` |
| 原始 W4 | 453.21 tok/s | 28.79 tok/s | 2 | 两轮后 `Running: 0` |

相对原始 W4，32K Prefill 提升 **25.57%**，Decode 变化 **+1.15%**；四个请求 usage 均为 `32256+128`，输出哈希一致。原始数据见 `paired-32k-ns.jsonl`。

## 48K 单轮交替补充

修正 metrics 连接关闭逻辑后，执行一轮 48K 配对：MindIEQ `521.49/25.34 tok/s`，原始 W4 `420.77/24.93 tok/s`（Prefill/Decode），Prefill 提升 **23.94%**，Decode 变化 **+1.66%**。两边 usage 均为 `49152+128`、输出哈希一致，容器日志最终均为 `Running: 0`。原始数据见 `paired-48k-ns-v3.jsonl`。

## 112K / 128-token 持续 Decode 对照

使用上述同一输入，输出扩大到 128 token，按 MindIEQ→原始 W4→原始 W4→MindIEQ 顺序完成两轮配对。两个服务已由前序长请求预热；本轮未额外计入专用 warmup。prefix cache 仍关闭，两个服务分别固定在不同芯片，因此存在芯片差异的潜在混杂，尚未做交换芯片对照。

| 服务 | Prefill 中位 | Decode 中位 | 总耗时中位 |
|---|---:|---:|---:|
| MindIEQ E5 | 378.69 tok/s | 16.87 tok/s | 310.07 s |
| 原始 W4 | 329.60 tok/s | 16.77 tok/s | 355.17 s |

Prefill 提升 **14.89%**，Decode 变化 **+0.63%**，总耗时减少 **12.70%**。四个请求均完成 `114560+128=114688` token，输出哈希一致，每次 Prometheus prefill count delta 为 1。最后两个服务均回到 `Running: 0`、`Waiting: 0`，日志显示 MTP 平均接受长度 3、两个位置接受率 100%。原始结果为 `paired-112k-fixed-o128-v1.jsonl`，包含原始阶段计时 delta。

复现时把上面的 `--output-tokens` 改为 `128`、`--repeats` 改为 `2`，并使用新的输出文件。128 token 的 Decode 阶段约 7.5 秒，本次仅支持“较 32-token 请求未观察到明显 Decode 回退”，不能替代长输出质量、业务文本 MTP 接受率、流式 TTFT/TPOT 或并发尾延迟验证。

## 32K 流式对照

在同一份 114560-token 离线样本上使用 `--prompt-limit 32256 --stream --output-tokens 128 --repeats 2`，保持已预热服务、prefix off，完成两轮交替配对。结果在 `paired-32k-stream-v1.jsonl`，包含逐文本事件时间、完整文本、usage、原始阶段 delta。输入是重复合成 benchmark 文本，不是会议摘要质量测试。

| 服务 | Prefill 中位 | Decode 中位 | 首非空文本 TTFT 中位 | 平均 TPOT 中位 |
|---|---:|---:|---:|---:|
| MindIEQ | 546.50 tok/s | 29.21 tok/s | 59.051 s | 34.125 ms |
| 原始 W4 | 451.76 tok/s | 28.82 tok/s | 71.420 s | 34.623 ms |

四次输出哈希一致，TTFT 减少约 12.37 秒。每次 128 个输出 token 以 44 个非空文本事件发送，因此事件间隔不是逐 token TPOT；本文的平均 TPOT 为 `(最终usage到达时间 - 首文本到达时间)/(completion_tokens-1)`，包含 HTTP/收尾开销，不能计算为逐 token P95。TTFT 是首个非空文本事件到达时间，若首 token 不产生可见文本，则不等于引擎首 token 时间。

流式采集器在最终 usage 事件后关闭响应，避免实验镜像不及时发送 DONE/EOF 时把连接等待算入延迟；不请求 continuous usage。离线模拟已验证无 DONE 的完整最终 usage 可以正常结束。此改动只影响 benchmark 客户端。

## 112K 流式近上限对照

在上述同一输入、预热服务、prefix off 下，使用 `--stream --output-tokens 128 --repeats 1` 完成一次配对。两次 usage 均为 `114560+128=114688`，输出哈希相同，各有44个非空文本事件，两个服务最后恢复空闲且健康检查为200。

| 服务 | Prefill | Decode | 首非空文本 TTFT | 平均 TPOT | 总耗时 |
|---|---:|---:|---:|---:|---:|
| MindIEQ | 378.50 tok/s | 16.88 tok/s | 302.733 s | 58.935 ms | 310.218 s |
| 原始 W4 | 329.59 tok/s | 16.74 tok/s | 347.651 s | 59.494 ms | 355.207 s |

MindIEQ 首文本等待减少44.918秒。原始数据 `paired-112k-stream-v1.jsonl` 保留逐事件时间、usage和阶段delta。这是一对样本，不能报告P50/P95；其作用是确认长流式响应可完整返回且TTFT与Prefill阶段计时一致。模型在该重复合成输入上续写相同文本，MTP接受率不能外推到业务摘要。

产品体验边界：112K冷输入仍需约5分钟才开始输出；即使MindIEQ有稳定方向性收益，也不能称为低延迟长上下文。下一步应评估真实会议文本的摘要/结构化提取质量和MTP接受率，再观察prefix缓存或增量输入减少冷Prefill的实际收益。

## 32K 合成会议决议更新 canary

该样本把三条有依赖关系的正式决议放在约10%、50%、90%的输入位置，并用中文普通讨论填充到32256 token。A给出初始预算/负责人/日期，B只更新预算/负责人，C更新日期和状态；模型需要合并最新字段并返回JSON。两个服务均正确返回 `{"budget_yuan":180000,"owner":"周启","deadline":"2026-11-12","status":"延期上线"}`，自然结束时各生成54 token。

| 服务 | Prefill阶段 | Decode阶段 | draft token 接受率 |
|---|---:|---:|---:|
| MindIEQ | 66.303s | 1.840s | 36/36（100%） |
| 原始 W4 | 76.475s | 1.875s | 36/36（100%） |

原始记录 `meeting-update-32k-v1.jsonl`，输入 `meeting-update-32k-v1.prompt.json.gz`。这只有一个合成任务，正确结果不代表完整会议摘要质量；两边也都使用W4+C8，不能证明相对BF16/W8A8的量化质量无损。该中文输入的Prefill速率与英文重复文本不同，不能将不同prompt的数字直接相减作为算子收益。

历史32K记录的config错误写入 `max_output_tokens=256`，实际请求代码使用128，服务自然结束54 token；本次保留原始文件不改写结果，并修正工具元数据。新的工具同时记录finish_reason。

复现（两个端点均需独占、prefix off）：

```bash
python3 tools/check_meeting_long_context.py \
 --mindieq-url http://127.0.0.1:18024 --original-url http://127.0.0.1:18028 \
 --model qwen36-w4a8-c8 \
 --prompt-file results/qwen36/mindieq-long-paired-20261014/meeting-update-32k-v1.prompt.json.gz \
 --out /tmp/meeting-32k-recheck.jsonl
```

这条质量请求允许EOS并限制最多128输出token，与性能基准强制输出128 token的协议不同；不混合统计两类Decode吞吐。

## 112K 合成会议决议更新 canary

三条决议位于114560-token输入的11458/57239/103023位置，要求合并最终预算、负责人、日期和状态。MindIEQ 与原 W4 均正确返回相同 JSON，各输出54 token；MTP均为18 drafts/36 drafted/36 accepted。输出未达到128-token上限，未见服务失败；两端完成后Running/Waiting均为0，health均为200。原始计数见 `meeting-update-112k-v1.jsonl`，输入见同名 `.prompt.json.gz`。

| 服务 | Prefill阶段 | Prefill | Decode阶段 | Decode | 总耗时 |
|---|---:|---:|---:|---:|---:|
| MindIEQ | 332.363s | 344.68 tok/s | 3.214s | 16.49 tok/s | 335.593s |
| 原 W4 | 368.284s | 311.06 tok/s | 3.243s | 16.34 tok/s | 371.542s |

该中文canary的Prefill收益约10.8%，低于重复合成吞吐样本的14.9%；它说明输入内容会影响路由和性能。单个canary只验证跨段更新提取，不替代真实会议摘要、广泛质量评测或P95。远端运行脚本尚未包含后来添加的finish_reason字段；本轮日志未记录该字段，不能据此报告精确停止原因。

```bash
python3 tools/check_meeting_long_context.py \
  --mindieq-url http://127.0.0.1:18024 \
  --original-url http://127.0.0.1:18028 --model qwen36-w4a8-c8 \
  --prompt-file results/qwen36/mindieq-long-paired-20261014/meeting-update-112k-v1.jsonl.prompt.json.gz \
  --out /tmp/meeting-update-112k-repeat.jsonl
```
