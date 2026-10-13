# MindIEQ 服务 B/A/B 对照（2026-10-14）

本轮检验上一轮native布局Prefill收益的可复现性及8K Decode差异。顺序B0（当前MindIEQ服务）→A1（重启原native服务）→B2（重启MindIEQ服务）；保持同一checkpoint、C8、MTP2、GDN、Graph、4096/512、112K、单流、prefix off。B0是已有暖服务，A1/B2重启后均warmup；顺序效应仍不能完全排除。

每轮8K与32256-token输入/128输出，warmup1/repeats3、每请求不同前缀，同输入SHA逐条检查。新增--spec-metrics保存每请求MTP草稿/接受计数，计算接受率与平均接受长度；这些是服务counter差值，非模型统计推断或纯kernel耗时。B0/A1另外同seed310随机64题、512输出上限，记录所有输出和截断；64题样本与此前32题集合不同，不能直接把两个总分比较。B2只复测性能。

执行脚本run-paired.sh依次停止/重启实验容器；必须确认芯片0独占、现有B0服务确实开启MindIEQ且health200后运行。脚本在固定新目录记录state.txt，不覆盖已存在数据；若健康超时，查看现有容器和日志，不盲目重启。实际启动PID与日志位于a2 mindieq-r3-20261013/paired-controller-20261014.{pid,txt}。制品hash沿用前轮，新增脚本来源见artifacts.json。start-baseline.sh/start-mindieq.sh为运行路径快照。

当前记录是进行中的实验，不宣称A/B门禁已经通过。完整结果同步后运行python3 results/mindieq-paired-20261014/analyze.py；分析器拒绝缺失/不完整结果、无效请求及输入SHA或质量协议不同的比较。

B0性能已完成：8K Prefill中位630.356、Decode35.797tok/s；逐请求接受率91.1%/100%/85.1%。32K Prefill550.744、Decode29.032tok/s，三次接受率100%。与上一轮MindIEQ结果接近；A1/B2未完成，不能宣称交替收益验证通过。B0的64题评估仍运行中，未提交其不完整输出。
