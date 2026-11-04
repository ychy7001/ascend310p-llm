# Ascend 310P 采集方法与算子证据

## 采集与解释口径

Profile独占设备，普通计时不启动Profiler；记录HTTP start/stop、首输出、warmup/capture边界及rank映射。Decode固定seed0、独立salt和整个请求prefix hits增量0。采集后以短题及A/B/A检查状态恢复。

分析器支持torch-npu `Start Time(us)`，同时报告跨stream kernel区间并集与实际捕获跨度，累计kernel求和不当作墙钟延迟或关键路径。客户端使用短单元tokenize后重复精确token数，收到最终usage后关闭流。

生产post2 daemon不支持在线解析；保留FRAMEWORK/PROF后使用同镜像`torch_npu.profiler.profiler.analyse`离线导出。无设备容器曾报`Init platform by driver failed`，补齐设备/驱动元数据访问后成功。两范围×双rank的4份非空CSV均可解析，未重跑普通基准。Stop时RECORD提示及HTTP控制边界可能造成少量步骤未捕获，窗口耗时只作诊断。

## 历史证据适用范围

[qwen36/rc-ep-bottleneck.md](qwen36/rc-ep-bottleneck.md)中的RC/C8/MTP1窗口：MatMulV2 45.66%、W4 12.25%、NotEqual7.67%、TransData7.11%、PA mask6.93%、C8转换3.99%、GDN3.11%。这些是捕获kernel累计时间占比，不是端到端墙钟占比，也不是a2 C16配置排名。

同文长上下文C8/MTP2捕获窗口在114560/128下PA mask36.77%、C8转换21.78%、MatmulWeightNz20.23%、W4 6.79%。当前a2保留配置为C16，不能继续把C8转换列为它的首要瓶颈。

旧Profile有停止导出失败 `ERR04005` 的记录；一次请求成功不代表取得可用Profile。新采集需验证CSV/trace可解析，采集后恢复普通计时，再跑正确性门禁。

## W4预算支线：C16

原始kernel CSV（gzip）、HTTP采集范围、前后quality、kernel摘要及服务日志：`results/profile-c16-20261103`。三份均导出成功，前后短题8/8与A/B/A通过。

| 捕获范围 | kernel条数 | 捕获设备跨度 | kernel区间并集覆盖 | 主要累计耗时 |
|---|---:|---:|---:|---|
| 4K Prefill，输出1 | 23248 | 6.437s | 98.79% | MindIEQ grouped 52.91%；MatmulWeightNz5.49%；GDN ComputeWy5.11% |
| 4K完整128输出 | 235239 | 10.475s | 94.72% | MindIEQ33.99%；MatmulWeightNz21.95%；native W4 5.43% |
| 32K首个输出后Decode | 167239 | 3.987s | 90.16% | MatmulWeightNz40.00%；PA mask21.62%；native W4 13.27%；TransData5.08% |

32K Decode中lm_head对应NZ矩阵形状`128,1,16,16;128,15520,16,16`，123次累计706372us，占捕获kernel求和约19.46%；这只是当前W4/MTP2窗口的shape归因，不是生产W8份额，也不自动证明INT8 head有端到端收益。GDN状态已FP16。

Profiler给出Stop时仍在RECORD的提示，窗口从HTTP控制边界采集；32K Decode可能漏掉启动控制返回前的少量步。三份kernel计数、范围与导出均可复核，Profile计时不参与常规吞吐。

## W8 TP2生产副本

生产`chunk_gated_delta_rule.py`的WY为Torch逐row循环、clone、sum及布局变换，FwdH/FwdO为AscendC。历史Graph[1,2,3]日志含EL0008/IPC notify事件资源不足；旧上游issue1565只能作线索，不能直接归因本轮Graph筛选。

原始CSV压缩、双rank摘要、采集HTTP范围及映射位于[production-w8a8-20261103](../results/production-w8a8-20261103/README.md)；profile-artifacts.json明确每份摘要对应的rank/范围。下表均为**单rank累计kernel耗时占比**，不能作为墙钟收益上限或关键路径占比。

| 范围 | rank0 / rank1条数 | 设备跨度 | kernel区间覆盖 | 主要累计耗时 |
|---|---:|---:|---:|---|
| 4K Prefill，输出1 | 38072 / 38072 | 3.162 / 3.161s | 97.45% / 97.51% | W8 MoE grouped matmul 17.78% / 17.91%；Slice 7.15% / 7.15%；Tril 5.52% / 5.47%；ViewCopy 5.31% / 5.24%；ReduceSum 4.98% / 4.95% |
| 32K首个输出后Decode | 380000 / 380000 | 4.466 / 4.464s | 90.44% / 91.07% | W8 MoE grouped matmul 23.15% / 22.79%；主要FP16 NZ MatMul变体20.11% / 19.90%；PA mask12.49% / 12.44%；lm_head对应MatMul4 9.11% / 9.07% |

Decode lm_head的NZ形状`128,1,16,16;128,7760,16,16`，每rank125次；其余MatMulV2变体合计24.62% / 24.36%，大部分Graph内shape为N/A，尚不能逐层归因。所有TransData变体合计5.85% / 5.70%；recurrent GDN约2.09% / 2.18%，FP16 state已启用。

Prefill的Tril/Slice/ViewCopy/ReduceSum与已归档WY Torch分解路径相符，是复用原生WY的优先调查线索；这些通用kernel也可用于其他路径，不能把它们全部求和称为精确GDN份额。W8 MoE算子和Decode FP16投影/输出头、PA mask是后续主要对照对象；GDN Decode并非当前最大的单项。

当前捕获C1，不能证明三并发的host开销或Graph收益；需候选实测。HCCL在Graph内的完整任务归因尚未做，kernel CSV中可见的hcom耗时不足以排除通信瓶颈。32K窗口捕获约125个decode步骤，HTTP start控制前的少量步骤未包含；Stop时仍有RECORD提示，计时只作诊断。

## WY-only候选Profile

独立采集在三轮业务复测之后进行，四份CSV/摘要/gzip见[production-wy-batching-recheck-20261103](../results/production-wy-batching-recheck-20261103/README.md)。4K Prefill每rank kernel38072→8492，捕获跨度约3.162→2.056s，覆盖率约97.5%。ComputeWy各60次、累计122.5/129.8ms（各rank kernel求和的6.08%/6.43%）；FwdH约91ms、FwdO约134ms，接近基线。Tril120→0、ReduceSum3860→160、ViewCopy3780→8；通用算子仍有其他使用，未把其全部归为GDN。该trace确认原生WY实际执行且分解调用减少，但其墙钟不是普通吞吐测量。

32K Decode仍每rank380000 kernels，跨度约4.505～4.507s，MoE/MatMul/PA-mask仍为主要项，符合单请求Decode无改善。数值拆分发现WY与FwdH接口稳定时FwdO原始输出仍会偏离，不能将WY提速当作数值修复，详细判定见results同目录adjudication.json。

## GDN数值诊断

门限固定为atol2e-4、rtol0.04、relativeRMSE<=0.02，独立CPU逐token递推参考；输出、末state、逐位稳定性分别记录。

111-token synthetic输入初测原Torch路径7/12通过、候选12/12；独占复核原路径3/12、候选12/12，偏差集中token56，末state稳定。独立无原生WY扩展的上游复现为5/12，属于不同运行，不能合并通过率。CPU生产chunk与独立递推输出最大差3.59e−6、状态5.22e−8。

完整链捕获显示WY/FwdH输出和FwdO输入稳定时raw FwdO仍可偏离。单独冻结FwdO为48/48；原算法独立重编译也48/48且逐位稳定，但完整Torch-WY调用历史18/24越界、该轮frozen-WY-clone为0/24。59份生产FwdO/FwdH/common源文件与生产Git blob一致。接口定位尚未证明同步、生命周期或内部kernel根因。

| 对照 | 数值结果 | 证据编号 |
|---|---|---|
| vector mask / 原算法加V/S fence | 各0/48，配对生产48/48 | `fwd-o-maskfix-fences-rejected-20261103` |
| PR17640新O/旧H | 0/48在容差、48/48逐位稳定；maxabs0.004327、relativeRMSE0.195875，异常token0～6、64～70 | `upstream-fwd-o-20261103` |
| executor cache默认 / `ACLNN_CACHE_LIMIT=0` | 21/48 / 25/48在容差，完整Torch-WY链均12/24越界 | `executor-cache-control-20261103` |
| post2 / post4完整栈 | recurrence probe 21/48 / 16/48；Torch-WY完整链15/24 / 14/24越界；frozen-WY-clone各1/24越界 | `gdn-runtime-control-20261104` |

post2/post4的Python chunk源码SHA同为`ab2b739bb6630ebe9fcb1d7af820f93adffce79668e32247bbb869119a0d6c69`；完整镜像差异不能归因为单独torch-npu，clone及关闭cache均未消除异常。候选取舍见[上游契约](research/gdn-upstream-pr-evaluation.md)，模型判定见[结果](ascend310p-results.md)。

PR17640 head18163f6b配套H/O在生产post2保留NZ状态binding、独立namespace/vendor。两卡×Torch/native WY四组，各24次完整链加24次冻结克隆，共192/192输出通过、逐位稳定，maxabs6.3153e−6。两个新进程各10/10严格用例（6项完整输出/状态、3项WY布局、1项CPU回退），输出maxabs8.2599e−6、state8.8640e−5。192次probe未逐次验证末state。

9.2.1默认与配套H/O的门限和扩大算子对照统一见[CANN实测](research/cann-upgrade-evaluation.md)。原生WY已在固定9.2.1重建：相同配套H/O，只切WY，两卡各15/15，共30/30，通过CPU FP32递推及每例3次bitwise重复；v1环境错误为0用例，v2通过。真实模型输入回放与固定token验收仍待做。证据：[native WY摘要](../results/native-wy921-20261104/README.md)。

## Dense启动诊断

启动阶段profiling/warmup错误不等于取得HTTP性能Profile。生产栈故障kernel为`QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21`，同步eager定位MLP down_proj；与ACL Graph同时设置`ASCEND_LAUNCH_BLOCKING=1`会先被配置校验拒绝。

首次down_proj为M4096/K8704/N5120，INT8激活ND、权重NZ转置view、F32通道/逐token scale；激活和逐token scale全零且finite。rank0/device0独立重放6项（全4096行的原/ND/NZ激活，原布局1/4/64行子集）通过CPU采样整数积，maxabs0。两TP分片×两逻辑设备交叉4项均通过，96元素采样maxabs0且输出finite；这是零输入启动复现，不是一般非零精度验收。

生产布局合成量化矩阵M=1/4096、两种MLP形状4项通过；另一NZ布局2项数值错、ND权重2项API拒绝，共8项。后续随机INT8权重加零激活/scale也能复现507015，无须checkpoint。

模型内重建down_proj物理[N,K] ND→新分配→NZ→transpose后INT8逐元素相等；两rank逻辑shape[8704,5120]、stride[1,8704]、format29、offset0、storage44564480B不变，rank0首调用通过、rank1仍507015。explicit expandable=True交叉为3PASS/1FAIL；rank1/device0交替5次：False4PASS/1FAIL，True3PASS/1FAIL，末次True初始化HDC等待128.88s后中止、未执行算子。

这些对照排除将零scale、单独分片/设备、权重重分配或allocator开关视为确定修复。完整运行栈及质量比较见[结果](ascend310p-results.md)，无权重最小复现见[上游草稿](upstream/310p-dense-w8a8-startup.md)。
