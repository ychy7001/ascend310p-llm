# C8 并发门禁（2026-10-14）

## 实验配置

- a2 chip1，独立端口 18025，chip0/18024 稳定服务未修改
- Qwen3.6-35B-A3B W4A8，TP1
- 动态 C8、GDN PR #17640、MTP2
- `max_num_seqs=8`，`max_num_batched_tokens=4096`
- 为隔离并发影响使用 eager（无 Decode Graph）
- 512 token 输入、32 token 输出，并发 2，warmup 1

## 结果

服务启动和单请求 warmup 成功。并发 2 的第一轮请求在 C8 Attention adapter 处失败，两个请求均没有返回 usage，EngineCore 随后退出。核心错误为：

```text
RuntimeError: C8 requires real four-tensor storage and one request
```

这说明当前 C8 实现明确要求单请求的四张真实 KV storage；`max_num_seqs>1` 不能仅靠 vLLM 参数打开。并发 4/8 没有继续执行，因为并发 2 已经触发同一结构性门禁。

本次不是性能失败，也不能据此评价 W4A8 的通用并发能力；它证明的是当前 C8 adapter 的并发能力为 **未支持**。chip0 的 18024 服务在实验前后 `/health=200`，显存和进程未变化。完整启动、warmup 和 traceback 见 `service.log`。

## 工程结论

产品默认继续使用 `max_num_seqs=1`。如果要支持并发 2/4/8，需要先改 C8 adapter 的 storage/layout 管理，使每个 sequence 拥有独立的四张 cache tensor，或实现 batch 维度的设备端 C8 转换和 attention；不能把当前单请求实现通过启动参数扩展成并发后端。
