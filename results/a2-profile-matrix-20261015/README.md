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

首组MTP2/4096已完成，剩余三组按顺序执行，结论待四组完成。首组正式两次中位：8K Prefill632.66/Decode37.99 tok/s；32K Prefill550.75/Decode32.41 tok/s。均返回128 token，prefix命中零；8K接受率98.84%/100%，32K均100%。这些是基线复现，尚未证明优于其他参数。用`python3 summarize.py`从下载到本目录的组结果生成`summary.json`。本轮不更改已交付默认值。
