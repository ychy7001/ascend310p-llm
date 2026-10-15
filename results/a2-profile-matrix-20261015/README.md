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

四组测速全部完成，共16个正式请求；逐项核实usage、phase计数、prefix零命中和实际容器配置通过。正式两次中位如下（tok/s）：

| 组合 | 8K Prefill | 8K Decode | 32K Prefill | 32K Decode |
| --- | ---: | ---: | ---: | ---: |
| mtp1-b2048 | 635.08 | 31.15 | 555.76 | 26.30 |
| mtp1-b4096 | 639.07 | 31.00 | 554.35 | 26.33 |
| mtp2-b2048 | 625.76 | 36.44 | 550.61 | 32.40 |
| mtp2-b4096 | 632.66 | 37.99 | 550.75 | 32.41 |

## 参数结论

32K四组对应输出哈希一致，输入一致且接受率均100%，优先用于排除输出路径差异：

- MTP1下batch2048→4096：Prefill−0.25%、Decode+0.11%。
- MTP2下batch2048→4096：Prefill约+0.03%、Decode约+0.03%。
- 固定batch2048，MTP1→2：Prefill约−0.93%、Decode约+23.21%。
- 固定batch4096，MTP1→2：Prefill−0.65%、Decode+23.12%。

本轮不支持为2048/4096保留速度档差异。现有a2默认4096已有112K门禁，矩阵结束时暂维持4096，随后经用户授权统一候选改为2048（见[后续部署](../unified-2048-20261015/README.md)），不宣称两者谁更快或硬件必须哪一档；2048尚无本轮112K完整请求验收，也没有证明其节省多少峰值内存。clt的2048属于历史验证范围，不是已证实的必要硬件限制。

MTP2值得保留为a2候选：32K在两种batch下都显示约23% Decode提升；但合成重复输入的接受率偏高，不外推真实会议业务、P95或结构化输出。clt/RC MTP2仍需各自验证；不得据此覆盖已有交付参数。

## 输出与容量边界

8K对应输出存在差异（详见comparisons.json），不能把速度差异直接当作等质量收益。MTP2/2048的第一个样本接受率83.33%，第二个100%，因此其8K Decode变动不应单独归因为batch性能。32K四组对应输出均一致，也只证明这些固定样本。

四组启动均保持114688上限且Graph成功：MTP1日志报告191507 KV tokens，MTP2报告186236；相同MTP下两个batch的该数值相同。所有日志报告KV预算4.30GiB（halved for workspace），Graph约0.09GiB。它们是启动容量估算，不能用作本轮112K完整请求或最小显存预算验收，也没有证明batch2048降低实际峰值内存。

服务日志中的655381-token警告来自benchmark在`/tokenize`先构造长文本、随后截取目标token IDs；实际推理usage已核对为8192/32768输入及128输出，不是超限推理。

## 恢复与下一步

矩阵脚本已停止实验容器并启动原qwen36-a2.service，原参数C16/MTP2/4096未改。服务ready和恢复快照另行核对。后续优先使用会议业务输入比较MTP1/2接受率、速度和质量，再在clt/RC测试统一候选；无需仅因本轮小幅波动继续扩大2048/4096扫参。
