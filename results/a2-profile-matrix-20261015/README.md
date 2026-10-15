# a2启动参数2×2对照（2026-10-15）

目的：区分历史验证范围差异与实际性能/兼容性需要，判断是否有必要保留 MTP1/2、prefill batch2048/4096 两套部署参数。不是并发 batch 测试。

## 固定项与变量

- 设备：a2 device0，EP310P3；仅此设备切换候选，device1不变。
- 运行代码：`/home/ychy/services/qwen36/c16-20261015/repo`，commit `5322528`。
- W4A8/SSZ 同一权重与算子，C16，TP1，max_model_len114688，util0.95，max_num_seqs1，prefix off。
- Decode Graph、GDN、MindIEQ E5启用，expert chunk512，MindIEQ阈值128。
- 变量：MTP draft tokens 1/2 × max_num_batched_tokens 2048/4096；Graph capture随MTP合法shape为1,2或1,2,3。
- 顺序：MTP2/4096 → MTP1/2048 → MTP1/4096 → MTP2/2048。
- 每组8192/32768输入，各预热1次、正式2次；正式输出128，temperature0、ignore_eos。同位置使用相同输入token IDs；用独立cache salt验证prefix零命中。
- 阶段吞吐来自Prometheus request phase计数差，非纯kernel速度。另保留客户端TTFT/TPOT、输出、MTP接受率、实际容器参数、NPU状态和服务日志。

`run_matrix.py`在a2执行；退出时恢复`qwen36-a2.service`。远端原始目录：`/home/ychy/ascend-lab-runtime/a2-profile-matrix-20261015/`。复现前应另选新输出目录，脚本拒绝覆盖已有组目录。

## 判定边界

本轮先排除参数影响，两个正式样本仅用于筛选，不能宣称P95或业务质量。结果必须同时检查输入哈希、MTP接受率和输出差异；合成文本高接受率不能代表会议业务。若差异明显，再对优选组合补112K和业务门禁；若差异处于波动范围，应统一参数，避免无证据的host分支。

跨机器比较还受到EP/RC、CPU、运行时、同步/异步调度、KV预算和prefix状态影响；a2上的胜出组合不会自动成为RC/clt已验证组合。

## 状态

已完成2/4组，第3组MTP1/4096正在启动；结论待四组完成。用`python3 summarize.py`更新汇总。正式两次中位如下：

| 组合 | 8K Prefill | 8K Decode | 32K Prefill | 32K Decode |
| --- | ---: | ---: | ---: | ---: |
| mtp2-b4096 | 632.66 | 37.99 | 550.75 | 32.41 |
| mtp1-b2048 | 635.08 | 31.15 | 555.76 | 26.30 |

单位tok/s。所有请求有效，prefix命中零。8K输入哈希一致，但第1个正式样本输出不同，第2个相同；MTP1第1个样本接受率91.04%，MTP2为98.84%。该样本的速度差异混有输出路径/接受率影响，不能当作等质量收益。32K两个正式样本输出哈希均一致，接受率均100%。目前两组同时改变MTP和batch，尚不能分别归因。
