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

已完成3/4组，第4组MTP2/2048正在启动。正式两次中位如下（tok/s）：

| 组合 | 8K Prefill | 8K Decode | 32K Prefill | 32K Decode |
| --- | ---: | ---: | ---: | ---: |
| mtp1-b2048 | 635.08 | 31.15 | 555.76 | 26.30 |
| mtp1-b4096 | 639.07 | 31.00 | 554.35 | 26.33 |
| mtp2-b4096 | 632.66 | 37.99 | 550.75 | 32.41 |

所有请求有效，prefix命中零。同长度、同repeat输入哈希一致。

- MTP1下batch2048→4096：8K Prefill+0.63%、Decode−0.48%；32K Prefill−0.25%、Decode+0.11%。本轮两个正式样本不足以支持4096有稳定优势。
- 固定batch4096，32K MTP2相对MTP1：Prefill−0.65%、Decode+23.12%；两次对应输出哈希均一致，接受率均100%。这是合成重复输入的结果，尚不能外推业务MTP接受率。
- 8K MTP1/2048与MTP2/4096的第1个样本输出不同、第2个相同；MTP1/4096与MTP1/2048的两个样本输出均不同。接受率随输出路径变化，不能把8K速度差异直接算作等质量收益。
- 32K目前三组两个对应样本输出均一致。最后一组完成后，才能补齐MTP2下batch的独立比较。暂不更改已交付配置。

服务日志中的655381-token警告来自benchmark在`/tokenize`先构造长文本、随后截取目标token IDs；实际推理usage已核对为8192/32768输入及128输出，不是超限推理。
