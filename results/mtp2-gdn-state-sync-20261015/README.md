# RC MTP2 accepted-token 快照隔离

本轮只验证 CPU fallback 的状态推进读取当前 GPU accepted-count 快照。不能据此归因全部 MTP2 故障，也不能将 RC 结果推广到 EP：clt 的已有版本已包含该读取方式，仍存在 MTP2 质量失败。

| 候选 | 验收 | 结论 |
|---|---|---|
| RC MTP2 eager，快照补丁 | 短题8/8、A/B/A通过；32,256/128 warmup+2轮；长请求后短题8/8；32K会议四字段正确 | 本轮门禁通过，尚无112K验收 |
| 相同候选启用target/draft Graph | capture完成，health200；首个短题持续无返回，两次栈采样停在draft-token回传event同步 | 不进入吞吐测试，不可作为默认 |
| 保留快照补丁，target Graph + draft eager | capture完成（0.21 GiB），health200；首个短题180秒超时，两次栈采样仍停在相同event同步 | 仅关闭draft Graph不足以修复；不进入吞吐测试 |

eager 两次冷请求 Prefill 为507.11/509.80 tok/s，Decode为8.31/8.67 tok/s，draft接受率78.0%/83.3%，prefix hit增量均为0。会议canary Prefill442.76、Decode9.52 tok/s。重复文本吞吐和会议质量是不同输入，不互作对照。

等待栈是观测位置，不能证明event API或CPU copy本身出错。后续需在保留快照补丁的条件下隔离draft Graph，并进一步检查图内任务与copy stream依赖；不应删掉同步来绕过等待。

上述draft Graph隔离已执行，结果仍停滞。实际写回使用Ascend runner的
`_copy_draft_token_ids_to_cpu`：copy stream先`wait_stream(default_stream)`，再执行
non-blocking D2H copy并record event；读取继承vLLM基类的event synchronize。
因此不能将栈中的vLLM文件路径理解为调用了CUDA复制实现。调用源码摘录保存在
`orangepi/draft-copy-protocol-excerpt.txt`。下一步应通过分段event/日志区分
默认stream上的生产任务、跨stream等待及D2H copy，先定位未完成阶段，再选择修复；
现有证据不支持删除event或直接把RC/310P1认定为根因。

文件入口：

- `accepted-token-snapshot.patch`：本轮行为改动。
- `orangepi/retry-protocol.md`：参数和复现命令。
- `orangepi/candidate-runtime.sha256`：关键运行时校验。
- `orangepi/candidate-p32256-retry.jsonl`：warmup、两次计时、逐token返回和指标。
- `orangepi/candidate-quality-after-32k.json`：长请求后短题与复用检查。
- `orangepi/candidate-meeting-32256.json`：会议决议canary。
- `orangepi/graph-first-request-stack*.txt`、`graph-service.log.gz`：Graph停滞证据。

原稳定MTP1容器已重启恢复；启动恢复验收另存`restored-mtp1-quality.json`。候选容器停止保留，未替换默认源码。

draft-eager隔离结束后，再次恢复原MTP1服务并通过8/8及A/B/A复验，结果见
`orangepi/restored-after-draft-eager-quality.json`；仅稳定容器运行。
候选180秒超时的结构化记录见`orangepi/draft-eager-outcome.json`，
日志和两次栈采样均保留，未生成可用的候选吞吐数据。
