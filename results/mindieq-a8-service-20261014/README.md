# MindIEQ 自定义 A8 量化服务验证（2026-10-14）

C3在310P3单芯片/48G运行：TP1、W4A8、动态C8、MTP2、Decode Graph、GDN、batch4096、expert chunk512、max_model_len114688、gpu_memory_utilization0.95、max_num_seqs1、prefix cache关闭。大于16token的Routed Experts Prefill使用MindIEQ grouped，自定义FP32算术A8准备；小输入与Decode保留native。权重不复制第二份专家bank，MTP保留BF16。

日志确认custom_quant=True；模型权重日志27.288GB、C8 116块及224MiB scratch、Graph0.30GiB。固定门禁与业务补测均已完成，服务保留a2端口18024。仓库MindIEQ默认开关仍关闭。

## 同协议模型结果

下表使用warmup1、重复3次的中位数；8K/32K输入SHA逐一匹配同日A1参考。C3没有进行新的交替重启，不能沿用旧dynamic_quant的B/A/B结果证明新量化器跨启动可重复。

| 指标 | 原native后端A1 | C3 | 变化 |
|---|---:|---:|---:|
| 8K Prefill tok/s | 502.936 | 607.019 | +20.7% |
| 8K Decode tok/s | 37.038 | 34.734 | -6.2% |
| 8K MTP汇总接受率 | 100% | 92.96% | 下降 |
| 32K Prefill tok/s | 450.355 | 533.057 | +18.4% |
| 32K Decode tok/s | 29.106 | 28.953 | -0.52% |
| 32K MTP接受率 | 100% | 100% | 持平 |

输出128token，8K全文1/3相同、32K全文3/3相同。固定64题均63/64，数字答案变化0、全文一致26/64；唯一205共同受512输出上限截断，第711题C3正常回答6。该小子集不证明全面精度等价。

114560输入/14输出检索正确，输入SHA与历史原后端匹配，303.264秒 vs348.941秒，总耗时降低13.1%。长请求后sanity5/5。它是单次检索canary，不是112K持续Decode速度或最长可用上下文证明。

## 业务输入补测

使用tools/bench_mtp_chat.py，warmup1、重复3次，temperature0、thinking关闭、输出上限256。参考results/pr17640-mtp-20261013/mtp2-chat.jsonl为历史同配置原后端；各输入SHA匹配，未做交替重启。

| 场景/输入token | 原TTFT → C3（秒） | 原Decode → C3（tok/s） |
|---|---:|---:|
| 对话/45 | 0.396 → 0.466 | 31.316 → 31.547 |
| 会议/962 | 2.169 → 1.921 | 39.225 → 39.071 |
| 代码/30 | 0.338 → 0.408 | 39.305 → 38.620 |
| 分析/42 | 0.384 → 0.464 | 34.189 → 33.794 |

短输入TTFT增加约70～80毫秒，会议TTFT下降11.4%。会议输出长度147→145，均自然结束，全文有差异；其他三个场景均达到256上限，不能据此评价完整任务质量，也不能把总耗时差异全部归因于计算性能。全部流式事件、文本、MTP counters在chat.jsonl，统计见chat-comparison.json。

## 复现与工程结论

新库SHA256：246cc4656cf401cc01586628bf018c2132251d9e22db2fb93ad17ddba8685de0。原型代码、构建、18shape/异步/Graph与真实256专家完整MLP见[单算子记录](../mindieq-a8-quant-20261014/README.md)。来源与许可见NOTICE；外部MindIEQ类不复制进公开仓库。

start-service.sh、run-service-gates.sh保留实验机命令；部署需要替换work/code/release路径。新版库独立目录，旧产物不覆盖。runtime.json记录容器实际配置，artifacts.json记录制品hash，preflight.json记录启动前检查。

```bash
python3 results/mindieq-a8-service-20261014/analyze.py
python3 tools/check_source_snapshot.py docs/source-snapshot-20261013.json
```

分析脚本拒绝不完整结果并核对输入/质量协议，生成comparison.json与chat-comparison.json。此目录只发布输出、索引和数据集哈希，不包含原始GSM8K数据集。

值得继续作为可选长输入Prefill优化：新量化器消除了本子集旧候选新增截断，保持约18%～21% Prefill收益。但8K Decode仍下降6.2%，短输入TTFT有额外开销，暂不改变默认。下一步测试输入规模分流、新候选跨启动复测与更广质量，并定位剩余MTP接受率差异。上述收益不是W8A8同条件比较；TP2 C8、多并发、32G、多模态及最大上下文未在本轮验收。
