# Ascend 310P Profiling 状态

2026-11-03：已完成W4A8 TP1/C16/MTP2三份新Profile并通过前后门禁。用户随后明确W8A8 TP2为生产基线，以下W4结果只适用于预算方案；生产W8 TP2两范围×双rank的采集及离线导出已完成，结果如下。

## 旧证据的适用范围

[qwen36/rc-ep-bottleneck.md](qwen36/rc-ep-bottleneck.md)中的RC/C8/MTP1窗口：MatMulV2 45.66%、W4 12.25%、NotEqual7.67%、TransData7.11%、PA mask6.93%、C8转换3.99%、GDN3.11%。这些是捕获kernel累计时间占比，不是端到端墙钟占比，也不是a2 C16配置排名。

同文长上下文C8/MTP2捕获窗口在114560/128下PA mask36.77%、C8转换21.78%、MatmulWeightNz20.23%、W4 6.79%。当前a2保留配置为C16，不能继续把C8转换列为它的首要瓶颈。

旧Profile有停止导出失败 `ERR04005` 的记录；一次请求成功不代表取得可用Profile。新采集需验证CSV/trace可解析，采集后恢复普通计时，再跑正确性门禁。

## 新采集计划

1. 同一固定C16镜像：4K与32K Prefill、生成稳定窗口，分别明确warmup/capture是否包括在采集中。
2. 分离target forward、draft forward、verify attention、GDN回滚/状态索引、rejection sampler、HCCL、图replay和host调度。
3. 对MatMul/W4按真实shape与token分布分组；检查GDN状态tensor真实dtype与layout，不能只读config。
4. 重点检查实际安装gdn_310.py中ragged state indices的CPU拷贝分支；它存在不等于当前每步执行，需要调用次数和时间线证据。
5. 对照MTP0/1/2、C1/C4，输出kernel和API求和、stream间空洞、同步等待、launch数与非profile墙钟，避免把重叠kernel和直接当延迟。

Profiler安装存在已确认；容器内torch-npu profiler与CANN导出可用性需先做短采集门禁。无新Profile前不开展大规模底层kernel修改。

## 工具准备增量

Profile入口派生自固定镜像，只增加显式Profiler/MTP开关，基线完成后顺序执行，不与计时请求共享NPU。已修复分析器未识别torch-npu `Start Time(us)`列的问题，新增跨stream kernel区间并集与实际捕获跨度统计，避免把重叠耗时求和当占用率；两种时间列和重叠区间的回归检查通过。Profile客户端改为短单元tokenize后重复精确token数，并在最终usage到达后关闭流，避免长tokenize警告及usage后不发DONE时等待。

## W4预算方案：新C16结果

原始kernel CSV（gzip）、HTTP采集范围、前后quality、kernel摘要及服务日志：results/profile-c16-20261103/。三份均导出成功，前后短题8/8与A/B/A通过。

| 捕获范围 | kernel条数 | 捕获设备跨度 | kernel区间并集覆盖 | 主要累计耗时 |
|---|---:|---:|---:|---|
| 4K Prefill，输出1 | 23248 | 6.437s | 98.79% | MindIEQ grouped 52.91%；MatmulWeightNz5.49%；GDN ComputeWy5.11% |
| 4K完整128输出 | 235239 | 10.475s | 94.72% | MindIEQ33.99%；MatmulWeightNz21.95%；native W4 5.43% |
| 32K首个输出后Decode | 167239 | 3.987s | 90.16% | MatmulWeightNz40.00%；PA mask21.62%；native W4 13.27%；TransData5.08% |

32K Decode中lm_head对应NZ矩阵形状`128,1,16,16;128,15520,16,16`，123次累计706372us，占捕获kernel求和约19.46%；这只是当前W4/MTP2窗口的shape归因，不是生产W8份额，也不自动证明INT8 head有端到端收益。GDN状态已FP16。

Profiler给出Stop时仍在RECORD的提示，窗口从HTTP控制边界采集；32K Decode可能漏掉启动控制返回前的少量步。三份kernel计数、范围与导出均可复核，Profile计时不参与常规吞吐。

## W8 TP2待验证线索

从历史Graph[1,2,3]原日志提取EL0008/IPC notify事件资源不足，见results/production-a1-audit-20261103/historical-graph123-resource-error.txt。另保存上游issue1565与评论：其中2025年旧版本讨论提到TP多rank时Graph数量受event-id资源限制；评论引用的旧文档不是当前栈能力保证。当前生产88eac271/torch-npu.post2/驱动26.0.rc1，与a2副本驱动26.1.1不同；先测量再归因。

生产三并发Graph[1,2]不能覆盖超出max capture2的decode batch，但这不直接证明host调度是主要瓶颈。Graph[1,4]保持两张图的筛选会在新W8 Profile后执行，实际capture/replay、HCCL、padding状态恢复和业务墙钟都需要验证。

W8采集入口的Decode请求现也固定seed=0、独立cache salt，并记录整个独占请求区间的缓存命中增量，要求为0；Prefill已有相同隔离。该变更只作用于尚未执行的诊断采集，不改变先前已完成的常规计时。

已归档生产镜像的`chunk_gated_delta_rule.py`：WY仍为Torch分解，含逐row循环、clone、sum和布局变换；FwdH/FwdO使用已有AscendC。这个源码事实使上游原生WY候选值得后续对照，但需要新W8 Profile确认其占比，不能从W4原生WY结果反推。原文件位于results/production-a1-audit-20261103/source/。


## W8双rank采集与离线导出

4K Prefill与32K Decode两个HTTP采集均完成，采集后8/8短题及A/B/A门禁通过。生产torch-npu.post2日志明确报告daemon进程不能在线解析，建议`torch_npu.profiler.profiler.analyse`离线处理；原始FRAMEWORK/PROF目录已保存。最初控制器因缺少kernel CSV标记失败，保留status-profile-export-pending.json，随后尝试同镜像/CANN、无模型与NPU设备的CPU容器离线导出（该尝试失败，修复过程见下文）。不能在CSV可解析之前声称取得W8算子排名。后续入口已加入该导出步骤与2范围×2rank共4份非空CSV检查。

离线导出问题已修复：完全不挂设备时msprof报告`Init platform by driver failed`；补齐同机设备/驱动元数据访问后，同镜像离线analyse成功，四份非空CSV均可解析。已保留失败日志、实际导出命令与恢复后的status；没有重跑普通基准或模型采集。

## W8 TP2当前算子证据

原始CSV压缩、双rank摘要、采集HTTP范围及映射位于results/production-w8a8-20261103/；profile-artifacts.json明确每份摘要对应的rank/范围。下表均为**单rank累计kernel耗时占比**，不能作为墙钟收益上限或关键路径占比。

| 范围 | rank0 / rank1条数 | 设备跨度 | kernel区间覆盖 | 主要累计耗时 |
|---|---:|---:|---:|---|
| 4K Prefill，输出1 | 38072 / 38072 | 3.162 / 3.161s | 97.45% / 97.51% | W8 MoE grouped matmul 17.78% / 17.91%；Slice 7.15% / 7.15%；Tril 5.52% / 5.47%；ViewCopy 5.31% / 5.24%；ReduceSum 4.98% / 4.95% |
| 32K首个输出后Decode | 380000 / 380000 | 4.466 / 4.464s | 90.44% / 91.07% | W8 MoE grouped matmul 23.15% / 22.79%；主要FP16 NZ MatMul变体20.11% / 19.90%；PA mask12.49% / 12.44%；lm_head对应MatMul4 9.11% / 9.07% |

Decode lm_head的NZ形状`128,1,16,16;128,7760,16,16`，每rank125次；其余MatMulV2变体合计24.62% / 24.36%，大部分Graph内shape为N/A，尚不能逐层归因。所有TransData变体合计5.85% / 5.70%；recurrent GDN约2.09% / 2.18%，FP16 state已启用。

Prefill的Tril/Slice/ViewCopy/ReduceSum与已归档WY Torch分解路径相符，是复用原生WY的优先调查线索；这些通用kernel也可用于其他路径，不能把它们全部求和称为精确GDN份额。W8 MoE算子和Decode FP16投影/输出头、PA mask是后续主要对照对象；GDN Decode并非当前最大的单项。

当前捕获C1，不能证明三并发的host开销或Graph收益；需候选实测。HCCL在Graph内的完整任务归因尚未做，kernel CSV中可见的hcom耗时不足以排除通信瓶颈。32K窗口捕获约125个decode步骤，HTTP start控制前的少量步骤未包含；Stop时仍有RECORD提示，计时只作诊断。


## WY 单变量实验进展（2026-11-03）

PR17640 固定 head18163f6b 的 WY-only 接入已构建、数值门禁与模型初轮完成，FwdH/FwdO保持生产版本；详见[结果](ascend310p-results.md)。四档 Prefill 提升38%～54%，长输出业务回放耗时减少约3.1%，Decode单请求未改善。会议4曾出现严重负责人错配，目前未验收；重启后串行6次映射正确，正在复测原三并发序列，不能据此消除首次失败。

111-token 原路径偏差已在空闲NPU复现：原路径3/12在固定门限内、WY候选12/12。CPU两参考在同一失败输入上的输出最大差3.59e−6。当前尚未定位偏差是在WY helper还是后续FwdH/FwdO；计划同步截取helper输出并固定其输入重放，避免仅以候选不再出现症状认定已修复。候选Profile安排在普通计时与业务复测之后，当前尚无候选kernel trace结论。

用户已明确a2为实验机，授权暂停临时unit独占实验并在结束恢复。a1未修改。实验目录 production-native-wy、production-wy-control-recheck、production-wy-batching-recheck；控制器顺序执行，不并发压测。

### WY候选双rank Profile已完成

独立采集在三轮业务复测之后进行，四份CSV/摘要/gzip见results/production-wy-batching-recheck-20261103/。4K Prefill每rank kernel38072→8492，捕获跨度约3.162→2.056s，覆盖率约97.5%。ComputeWy各60次、累计122.5/129.8ms（各rank kernel求和的6.08%/6.43%）；FwdH约91ms、FwdO约134ms，接近基线。Tril120→0、ReduceSum3860→160、ViewCopy3780→8；通用算子仍有其他使用，未把其全部归为GDN。该trace确认原生WY实际执行且分解调用减少，但其墙钟不是普通吞吐测量。

32K Decode仍每rank380000 kernels，跨度约4.505～4.507s，MoE/MatMul/PA-mask仍为主要项，符合单请求Decode无改善。数值拆分发现WY与FwdH接口稳定时FwdO原始输出仍会偏离，不能将WY提速当作数值修复，详细判定见results同目录adjudication.json。


## 算子诊断与启动故障记录的区别

FwdO固定输入/完整链重复、独立重编译、两种mask补丁、现有上游FwdO单独复用及executor-cache对照属于数值诊断，不是普通吞吐或新Profile。27B Dense在启动profiling阶段量化Matmul报507015，没有获得HTTP性能Profile。真实输入重放6项通过只说明独立进程中的数值检查，不证明模型内调用顺序正确。保持此前35B-A3B双rank Prefill/Decode四份Profile证据不变，具体新诊断见结果文档。
