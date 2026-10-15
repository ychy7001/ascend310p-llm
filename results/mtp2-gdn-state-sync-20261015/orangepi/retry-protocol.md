# RC accepted-token 快照候选：32K 重测

- 容器：`qwen36-mtp2-state-sync-32k-20261015`，与稳定服务独占运行。
- TP1 / W4A8 / C16 / MTP2 / eager / GDN / MindIEQ / 同步调度。
- `MAX_MODEL_LEN=32768`、batch 2048、专家 chunk 512、KV 3 GiB。
- 仅覆盖 `patch_mamba_utils.py` 的 accepted-token 快照读取；其他候选文件和算子校验见 `candidate-runtime.sha256`。
- 短片段分词后重复 token ID，精确构造 32,256-token 输入，给 128-token 输出预留空间。
- warmup 1 次、测量 2 次，每次独立 cache salt，要求 prefix cache hit 增量为零。
- 请求阶段计时来自独占服务 Prometheus 计数，不是纯 kernel 耗时。
- 这是合成重复文本吞吐探针；不能代替会议质量测试。

```bash
python3 bench_server_phases-fixed.py \
  --base-url http://127.0.0.1:18024 --model qwen36-w4a8 \
  --out candidate-p32256-retry.jsonl \
  --prompt-tokens 32256 --output-tokens 128 --repeats 2 \
  --unique-prompts --spec-metrics --cache-salt-prefix mtp2-snapshot-retry
```

此前一次启动因稳定服务已运行、可用内存不足而失败；这是实验隔离问题，不作为 MTP2 算子失败证据。
