# RC MTP2 完整 Graph：长上下文与匹配 MTP1 对照

本轮以 `6c44228` 的 32K 全 Graph 候选为起点。独立创建 MTP2、MTP1
两容器，顺序独占 NPU，原稳定容器保留，异常或完成后自动恢复并复验。
`run.py` 是实际主机编排脚本，不属于部署运行时。

共同参数：TP1、C16、W4A8、GDN、MindIEQ、同步调度、target/draft Graph、
max_model_len=114688、max_num_batched_tokens=2048、expert chunk512、
KV 逻辑预算4GiB、prefix caching开、视觉开、max_num_seqs=1。
两者保留相同accepted-token快照与设备slot算术补丁，仅MTP=1/2不同。

原3GiB MTP2池在32K配置报告容量约99103 token，因此本次改4GiB；
这只是逻辑预算，不是NPU实际分配量或48GiB整机可用性的证明。
RC为共享内存机器，内存结论必须结合实际cache storage、工作区和主机RSS。

每个候选依次执行：短题8/8+A/B/A、32K会议canary、32K和114176输入/128输出
各warmup1+冷请求2、114176会议canary、短质量复验。会议输入只生成一次，
对照复用相同token-ID文件；该canary不是正式精度评测。性能脚本固定采用仓库
`tools/bench_server_phases.py`（短片段分词后重复token-ID），每轮独立cache salt，
核对prefix hit增量0和请求哈希。保留完整原始JSONL，不依赖容器内旧脚本。

当前进度以主机`state.txt`和`run.log`为准，尚未完成的测量不得填成结果。

## 运行中调整与失败记录

首次创建脚本遗漏host IPC/PID，在模型加载阶段发现后停止，未产生测量，
原容器改名privateipc-preflight保留；正式两候选均核对为host IPC/PID。

MTP2正式候选短题8/8、A/B/A、32K会议四字段通过。112K配置下32K
Prefill507.82/509.77、Decode16.36/16.82 tok/s，接受率42.75%/44.78%。
不能沿用上一轮max32768候选的23tok/s作为112K档指标。

归档复核确认：114176输入的warmup已在301.78秒完成，输出32token；随后计时请求被人工中断，没有完整计时轮次。14:41和14:43两次采样处于长Prefill，均在GDN状态读取，第355行；不能据此判定无进展。早期“warmup未完成”的观察已由最终原始JSONL纠正，不计作失败。
设备日志的PID95368 AI Core异常出现在停止时间附近（设备时间UTC+8），
不能确定是故障原因还是终止效应，不将它当作驱动版本缺陷的证明。

当时调整实验顺序，后续MTP1对照仅执行32K配对基准和
一次114176会议canary，显式使用`--mtp 1 --resume-startup --skip-long-bench`。
MTP1启动中的编排重载只修改实验流程，不重建容器或改变运行参数。

## 对中断判断的修正

MTP1 114176会议请求在335.38秒后正常完成，四字段正确，Prefill344.26、
短答案Decode14.58 tok/s，随后8/8+A/B/A通过。第二次采样堆栈从GDN切换到
MindIEQ，证明先前重复采到GDN读取并不意味着请求无进展。MTP2首次在相近
时间被人工停止，故只记“人工中断、未完成”，不能将其记为已证实挂起或
112K失败。`recheck_long.py`复用MTP1的114176输入，给完整1800秒请求超时，
仅补验该长canary及前后质量，结束恢复稳定服务。

32K配对已核对两轮输入哈希与完整输出完全一致，prefix hit0：
MTP1 Prefill511.84/513.49、Decode20.10/20.72；
MTP2 Prefill507.82/509.77、Decode16.36/16.82 tok/s。
两者草稿调用数69/67、接受数59/60完全相同，MTP2额外草稿没有贡献接受数；
服务日志第二位置接受率0，与约19%的Decode下降一致。只是2次顺序测量，
不是P95或随机交错的统计性能结论。

## 最终测量结果与限制

MTP2补验114176输入已完成：耗时337.00秒、Prefill341.89 tok/s、短答案Decode17.92 tok/s，输出54token。与MTP1的输入哈希及完整输出逐字一致，四字段正确；前后8/8及A/B/A一致且无泄漏。因此首次人工中断不记为112K失败，更不能用重复GDN堆栈证明挂起。

| 测试 | MTP1 Prefill/Decode tok/s | MTP2 Prefill/Decode tok/s |
|---|---|---|
| 32K冷请求轮1，128输出 | 511.84 / 20.10 | 507.82 / 16.36 |
| 32K冷请求轮2，128输出 | 513.49 / 20.72 | 509.77 / 16.82 |
| 32K会议canary | 446.35 / 21.32 | 442.62 / 17.90 |
| 114176会议canary，54输出 | 344.26 / 14.58 | 341.89 / 17.92 |

长会议Decode仅是短答案阶段计时，不是固定128/256输出长Decode基准。MTP1在32K对照后执行，MTP2补验在重新启动和短质量后执行，请求历史不同，不能把单次canary当作MTP2稳定提速证明。

两档实际独立cache storage均为8174960640字节（约7.61GiB），逻辑池4282122240字节，预算4GiB。不能把逻辑预算当作物理内存占用；本机共享内存测试不证明物理48GiB整机通过。已验证配置max114688、实际114176+54，不宣称更长上下文或并发通过。

结论：功能门禁推进到112K，但112K配置下32K配对Decode约下降19%，性能门禁未通过，RC默认仍保留原C16/MTP1/Graph/GDN/MindIEQ、4096/512、112K。诊断补丁暂不并入默认。下一步按单变量区分请求顺序、max_model_len、KV池及draft Graph，定位长Prefill/复用后第二草稿的slot/GDN状态/Graph元数据更新，不混做驱动升级。

复现：`run.py`是主机顺序编排脚本；实际MTP1续测用`python3 run.py --mtp 1 --resume-startup --skip-long-bench`，MTP2补验用`python3 recheck_long.py`，原始结果核对用`python3 summarize.py > summary.json`。脚本路径和容器名为本轮固定，重跑须新建独立记录目录和名称，不直接覆盖。正式部署不使用这些编排脚本。最终恢复验收以`restored-mtp1-quality.json`为准。

最终恢复确认：仅`qwen36-orangepi`运行，health200、8题8/8、A/B/A一致且无canary泄漏，候选容器均停止保留。见`restored-mtp1-quality.json`，状态`recheck-completed-restored`。
