# wy-5 高精度参考与 logprob 比较器可行性核查

结论：**待定，停止模型实验，未启动 a2 NPU 窗口**。

已修复并测试统一的 `tools/compare_token_logprobs.py`：

- top-k 截尾时目标 token 缺失，记录目标 logprob 差的下界，不中止比较。
- 两侧 top-k 没有共同 token 时，KL 记录为不可估计并列出位置，不抛异常。
- 噪声底本身含截尾时，不把下界当作精确阈值。候选实测值低于噪声下界可以判定非劣；候选也截尾或超过下界时保持待定。
- 判定倍率按本轮冻结协议为 `1.0`，无绝对阈值豁免；比较器同时保留原有 greedy 首分歧和共享 top-k 条件 KL 记录。

测试命令：

```bash
python3 -m unittest discover -s tools -p test_compare_token_logprobs.py
```

结果：**9/9**。

## 高精度参考 R

按用户规定的优先顺序核查：

1. 已有 GPU 35B 环境使用 GPTQ Int4/W4A16 权重，3090 当前只有约 0.8 GiB 可用显存，不能启动同模型 BF16 或更高精度参考。
2. 已有 GPU 27B 环境是另一模型、W4A16、FP8 KV、MTP3 和不同运行栈，不能充当 Qwen3.6-35B-A3B W8A8 TP2 的高精度 R。
3. 本机缓存没有同模型 BF16/FP16 权重；发现的 35B 文件是 GPTQ 量化 shard。只读 safetensors header 审计见 `raw/gptq-header-audit.json`：7 个 shard、124611 个 tensor、约 24.42 GB，量化配置为 `bits=4`、`group_size=128`、`method=gptq`、`pack_dtype=int32`。
4. 本机 CPU 没有可加载的同模型高精度 checkpoint；当前可用内存约 41 GiB、交换空间已基本耗尽，不具备安全运行 35B 高精度 teacher-forcing 的条件。没有新下载权重，没有访问 a1、yys 或本机 3080。

因此本轮不伪造 R，不把 GPU GPTQ 或旧模型输出当作高精度参考，也不启动 a2 NPU。

## wy-5 判定协议冻结内容

在查看任何 wy-5 B/B0 数据前，判定协议冻结为：

- 输入：复用 wy-2 冻结的 35 条 token ID 输入。
- R：同一输入逐位 teacher-forcing 高精度 logprob 分布；当前 R 不可取得，故模型门禁不执行。
- A、B、B0、R 各报 top-1 一致率、KL、最大目标 logprob 差和首分歧位置。
- 新采用条件：B→R 各指标不劣于 A→R，允许差值不超过 A′→R 与 A→R 的噪声底；保留旧 A/B/A′ 结果作为参考记录。
- B 使用 wy-4 修复版；B0 使用修复前原生 WY 制品。B/B0 均禁止回退到 Torch。
- G2 仍使用 GSM64、13 例结构化 C1/C3、8 短题前后，各项相对 P 和 A 的通过数差≤1。
- 性能仅在 G1、G2 均通过后运行，报告会议回放、RAG 8K/16K/32K、通用 1K/4K/16K/32K×C1/C4 和 WY 冷 Prefill。

## 真实限制

wy-4 已完成的 A/A′ 数据显示最大目标 logprob 差至少为 7.31，且 14/35 条 greedy 序列发生分歧；该噪声底必须由同轮数据重新完整记录，不能用绝对小阈值替代。由于本轮没有合格 R，未重跑 A/A′、未运行 B/B0、未运行 G2 和性能。后续需要提供同模型 BF16/更高精度 checkpoint，或释放合适的 GPU/CPU 资源后再继续。
