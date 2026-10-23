# MindIEQ 服务 B/A/B 对照（2026-10-14）

本轮检验上一轮native布局Prefill收益的可复现性及8K Decode差异。顺序B0（当前MindIEQ服务）→A1（重启原native服务）→B2（重启MindIEQ服务）；保持同一checkpoint、C8、MTP2、GDN、Graph、4096/512、112K、单流、prefix off。B0是已有暖服务，A1/B2重启后均warmup；顺序效应仍不能完全排除。

每轮8K与32256-token输入/128输出，warmup1/repeats3、每请求不同前缀，同输入SHA逐条检查。新增--spec-metrics保存每请求MTP草稿/接受计数，计算接受率与平均接受长度；这些是服务counter差值，非模型统计推断或纯kernel耗时。B0/A1另外同seed310随机64题、512输出上限，记录所有输出和截断；64题样本与此前32题集合不同，不能直接把两个总分比较。B2只复测性能。

执行脚本run-paired.sh依次停止/重启实验容器；必须确认芯片0独占、现有B0服务确实开启MindIEQ且health200后运行。脚本在固定新目录记录state.txt，不覆盖已存在数据；若健康超时，查看现有容器和日志，不盲目重启。实际启动PID与日志位于a2 mindieq-r3-20261013/paired-controller-20261014.{pid,txt}。制品hash沿用前轮，新增脚本来源见artifacts.json。start-baseline.sh/start-mindieq.sh为运行路径快照。

当前记录是进行中的实验，不宣称A/B门禁已经通过。完整结果同步后运行python3 results/qwen36/mindieq-paired-20261014/analyze.py；分析器拒绝缺失/不完整结果、无效请求及输入SHA或质量协议不同的比较。

B0性能已完成：8K Prefill中位630.356、Decode35.797tok/s；逐请求接受率91.1%/100%/85.1%。32K Prefill550.744、Decode29.032tok/s，三次接受率100%。与上一轮MindIEQ结果接近；A1/B2未完成，不能宣称交替收益验证通过。B0的64题评估仍运行中，未提交其不完整输出。


## 激活量化边界诊断（空闲芯片1）

check_activation_quantizers_310.py覆盖K512/2048、每种16行，包括零、常数、线性、一般随机及微小幅值；native pack输出在全部40960元素上与FP32除法/round/clamp[-127,127] CPU参考一致。默认FP16 dynamic_quant与native量化码共2912处不同，其中小幅值两行占2556处；其重建最大误差约6.90e-5/6.65e-5，接近该行最大幅值。大部分普通幅值差异较少，不能把小幅值结果外推为整体质量失败。

FP32输入尝试失败：aclnnDynamicQuantV2返回161002，明确支持dtype仅DT_FLOAT16；保留错误日志。诊断中FP16输入乘128再将返回scale除128后，小幅值量化码差异降至4/20处、重建误差约2.77e-7/2.73e-7；总差异380处。这个结果支持后续调查FP16动态量化的数值范围/舍入，尚未证明内部具体算法。固定倍数可能导致大输入溢出，未作为模型补丁采用；模型当前仍为原dynamic_quant实现。

首次诊断还出现镜像Triton导入、设备逻辑编号与自定义OPP查找错误，修正并保留失败日志。只挂载davinci1时在本环境使用ASCEND_RT_VISIBLE_DEVICES=0；测试并未使用服务芯片0。上述错误不属于量化数值失败。

B0固定64题完成62/64，两题205/711均512上限截断，未提取答案；A1同协议结果仍待完成，不能宣称质量门禁通过。

A1性能已完成：8K 502.936/37.038、32K450.355/29.106tok/s，接受率均100%。B0相对A1 Prefill+25.3%/+22.3%，Decode-3.35%/-0.25%；输入SHA逐条匹配，8K输出1/3一致、32K3/3一致。详见interim-performance.json；A1质量与B2仍未完成。


## 完整B0/A1/B2结果

| 条件 | A1原native | B0 MindIEQ | B2 MindIEQ |
|---|---:|---:|---:|
| 8K Prefill tok/s |502.936|630.356|628.648|
| 8K Decode tok/s |37.038|35.797|35.654|
| 8K汇总接受率 |100%|91.85%|91.85%|
| 32K Prefill tok/s |450.355|550.744|549.644|
| 32K Decode tok/s |29.106|29.032|29.097|
| 32K接受率 |100%|100%|100%|

两轮候选Prefill相对A1提升25.3%/25.0%（8K）、22.3%/22.0%（32K）。8K Decode下降3.35%/3.74%，同时接受率和输出有变化；只证明伴随关系，不证明全部因果。32K输出均3/3与A1全文一致，速度接近。分析器确认全部条件输入SHA相同。B0/A1质量62/64 vs63/64，新增711截断（原后端答案6）；全文22/64一致。B2未重复质量集，不声称跨启动质量复现。

结论：Prefill收益跨启动可复现，但当前dynamic_quant版本不通过“保持质量且无Decode退化”的默认采用门禁，继续保持实验开关默认off。下一阶段独立FP32算术A8原型见../mindieq-a8-quant-20261014。原始服务日志、stop/start状态与PID为历史快照，不应据此判断进程仍在运行。完成分析python3 results/qwen36/mindieq-paired-20261014/analyze.py，comparison.json为自动计算结果。
