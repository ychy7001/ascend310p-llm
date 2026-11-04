# 是否升级 CANN：2026-11-04 评估

结论：**有必要将CANN 9.2.1完整配套栈列为a2下一阶段优先对照；当前不升级a1，不在旧镜像中单独替换CANN。** 升级目的为验证基础栈兼容性及故障是否消失，暂未证实新版修复本轮GDN或Dense507015。此前只考虑#18057的9.2.0依赖不完整；应优先固定正式发行组合。

## 官方依据与实机证据

已通过官方GitHub API和固定tag原始文件查询并归档至 `results/cann-upgrade-audit-20261104/`：

- vLLM-Ascend v0.30.0于2026-10-29 14:07:04 UTC发布，API标记prerelease=false。
- 固定v0.30.0兼容矩阵为：vLLM/vLLM-Ascend均v0.30.0、CANN9.2.1、PyTorch2.10.0、torch-npu2.10.0.post7；不能随意混合版本。
- 固定v0.30.0的Dockerfile.310p使用 `quay.io/ascend/cann:9.2.1-310p-ubuntu22.04-py3.12`，构建默认vLLM_TAG=v0.30.0；目标存在明确310P构建路径。需固定最终镜像digest，不能凭tag认定本地已有同一版本。
- 查询时main安装文档列出Ascend HDK26.1.1、NNAL9.2.1及Toolkit+Ops9.2.1，明确Atlas300IDUO对应310p镜像。a2驱动version.info为26.1.1，与该驱动版本相符；**尚未独立核实固件版本及发行要求全部匹配**，不把此项写成完整HDK通过。
- 发行说明包含Qwen3.8-27B、混合线性注意力缓存、GDN prefill及310P量化部署相关修复。不能从这些概括推断本轮QuantBatchMatmulV3_NZ错误hash或共享FwdO偏差已修复，需分别重放合成复现及真实模型。
- 当前生产镜像CANN9.1.0-beta.1/post2；已测另一完整镜像CANN9.1.0/post4，GDN偏差仍在，Dense仅能启动但质量7/8。它不是9.2.1试验，也不证明9.2.1无效。

额外兼容点：v0.30.0的310P Dockerfile显式卸载triton-ascend/triton。通用兼容矩阵的Triton版本不能当作310P开箱支持的证明；#17604 Triton prepare依然只作算法参考。CANN9.2.1基础镜像也不自动证明包含#18057所需的外部ops-transformer固定commit及其Python API。

## 建议验证顺序与采用条件

1. 保留旧镜像、权重和启动清单；a1继续只读。核对a2固件及9.2.1发行硬件支持；优先取得或构建固定tag/digest的310P完整镜像，读取真实Toolkit、OPP、NNAL、torch-npu、vLLM及Ascend版本。构建原生算子时清理旧编译产物，使用独立缓存及vendor，避免9.1二进制混入。
2. 第一轮使用新版原生默认路径，关闭本实验WY/H/O/gating补丁；先跑无权重Dense量化复现与独立GDN输出/末state/重复性。失败就保留日志，不先叠加PR掩盖原因。旧vLLM版本与新CANN未在官方矩阵中配对，不用任意混装制造“仅CANN A/B”。整个官方栈对比若成功，只能先称栈升级有效，不能单独归功CANN。
3. 算子检查通过后，依次跑35B W8A8 TP2及27B Dense。先短请求、状态复用、事实与会议关系、同索引GSM，必要的长上下文/边界和逐token或logits检查；预期答案存在/HTTP200不等于语义正确。Dense要先得到质量合格的可复现基线。
4. 质量过关才复用此前相同TP2/C1、输入长度、输出长度、预热与中位数、零前缀命中协议测Prefill/Decode，记录内存及启动代价。新版默认路径若足够好，再决定是否需要#18057/WY；不先把所有PR叠上。

当前未安装9.2.1、未修改驱动/固件、未发布新生产镜像；上述为完成来源核对后的实验决策，不是已验证性能结论。a2先前质量候选已停止，原系统服务独立健康检查HTTP200，见 `results/production-gdn-pair-quality-20261104/service-restoration.json`。

## 用户确认后的执行进展

用户同意a2隔离验证，并提供9.2.1合一包、310p-ops、NNAL安装参考。已对官方OBS合一包作64KiB只读Range读取（HTTP206，完整包约2.11GB），安装器头部确实列出toolkit/driver/all功能选择，因此不在宿主机直接执行默认安装。未下载完整run包，未执行安装器。用户给的FAQ/指南正文返回站点脚本，未将该响应当作可读安装说明。

a2实测物理卡ID=8，两个逻辑310P3，驱动26.1.1、固件7.5.0.5.220，设备自身Compatibility=OK。该状态仅证明当前设备报告兼容，不替代9.2.1完整实测。已开始rootless拉取quay.io/ascend/vllm-ascend:v0.30.0-310p；现有1.2TB空闲，保留旧镜像。镜像拉取完成后冻结完整image ID/digest，先无NPU只读核对包版本，再暂停a2临时服务运行无权重GDN和Dense合成检查；控制器finally恢复原服务。

## 本轮实测最终结论

已完成a2容器侧升级验证：官方v0.30.0-310p镜像ID d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a，实际Toolkit/OPP/NNAL9.2.1、torch-npu2.10post7、vLLM0.30.0+empty、Ascend0.30.0。宿主保持原驱动/固件。

**目前不采用新版默认栈，GDN正确性门禁仍失败。** 初始T111两卡96/96通过，Dense合成seed310..314×allocatorFalse/True共10/10通过且本轮无507015；扩展默认GDN15组仅13/15通过。独立新进程T512/seed318复核，35B对应HV16为29/48在容差内，Dense对应HV24为0/48，末state两组48/48在容差且逐位稳定。最大输出abs约0.00504/0.00599，相对RMSE约0.0652/0.0749，原阈值仍为atol2e-4、rtol0.04、relativeRMSE<=0.02。不能把小形状通过概括为GDN修复。

35B模型启动被门禁阻止，Dense模型也未启动，本轮无整模型质量/性能结论。下一步保留9.2.1为隔离诊断环境，优先围绕T512真实输出偏差定位/验证H/O配套修复，不能靠放宽容差、跳过门禁或直接迁移生产。主机不需再次覆盖安装合一包。结果与脚本冻结见results/cann921-upgrade-20261104；新的无权重复现包results/upstream-gdn-cann921-repro-20261104.tar.gz，未公开发布。

## 2026-11-04：9.2.1配套H/O已通过扩大算子门禁

固定官方镜像ID d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a，使用同一工具链重新编译PR17640固定head18163f6b的成对H/O，独立vendor/namespace，保留原始NZ状态binding。**Torch WY保持默认**，没有加载旧9.1 WY二进制或改算法。

| 检查 | 新版默认栈 | 9.2.1配套H/O、Torch WY不变 |
|---|---|---|
| T512/seed318/HV16，device0 | 29/48在容差内，输出1/48等于首轮 | 48/48在容差内且输出/末state逐位稳定 |
| T512/seed318/HV24，device0 | 0/48在容差内，输出1/48等于首轮 | 48/48在容差内且输出/末state逐位稳定 |
| 同输入device1 | 本轮默认对照未覆盖 | HV16、HV24各48/48通过 |
| 15组严格边界 | device0为13/15 | 两卡各15/15，每组额外3次逐位重复 |

配套H/O的T512输出maxabs为6.54e−6/6.38e−6，relativeRMSE约0.000424；15组输出最大maxabs1.025e−5、末state1.2465e−4，原门限atol2e−4、rtol0.04、relativeRMSE<=0.02未改变。结果说明此配套组合能消除已测样本异常，尚未区分H或O根因，不是整模型质量/性能验收。

验证前纠正加载顺序：vLLM bootstrap会将默认vendor置前，而9.2.1默认库也导出相同Vllm API名；独立NPUBridge会在dlopen时捕获环境。故在bootstrap之后、dlopen之前重新前置paired vendor，并记录API/扩展SHA和实际映射库。控制器在未运行验证时暂停，保留旧脚本和调整记录；算法/编译源未改。6个验证进程均只映射配套libcust_opapi.so。原始通用probe头仍称default/original scope，实际实验范围以wrapper日志和adjudication为准，未修改原始jsonl。

证据：results/upstream-gdn-pair921-v1-20261104/。a2原服务已独立HTTP200恢复后，继续进入35B W8A8 TP2质量筛选，配置为9.2.1+paired H/O+原Torch WY；暂不做WY性能比较。


## 2026-11-04：9.2.1配套H/O＋原Torch WY的35B质量筛选完成

该组合通过算子门禁后实际启动Qwen3.6-35B-A3B W8A8 TP2，保持maxlen262144、batch4096、seqs4、Graph[1,2]、SSM FP16、MTP0、prefix on，利用率0.90。各rank权重18.8545GB；完整上下文配置通过准入，不代表已发送262K请求。独立检查两个worker均映射paired API库，并记录原默认库仍服务其他算子。

前后短题各8/8、ABA输出一致无canary泄漏，事实16/16，业务自动28/28（20会议、3tool、5schema），但人工语义仍不通过。会议4本轮周明/刘洋/何丽映射正确，却把未标明完成状态的当天回复写成“已于当天完成回复”；会议5同样编造已完成停用，并把行动项截止时间从“当天”写成“本周内（已完成）”。原生产A也出现前两项完成状态编造，不能当作本候选独有的数值回归；本周内是本轮额外差异，仍需配对复核。20会议均已人工阅读。

28项原始请求与原生产A逐项严格相同；25条非tool文本7条相同，文字变化不能直接等同质量或数值退化。最终not_accepted_business_semantics，无新GSM、性能和生产采用。首次启动NNAL nounset错误发生在权重加载前，修正后v2实际完成本轮；原失败保留，不计入模型质量。

P0结论需分开：已测GDN数值异常可以由现有配套H/O在Torch WY不变时消除；自然语言完成状态问题也存在于原模型基线，尚未证明是GDN/CANN引起。下一步补齐负责人/截止时间/执行状态的同输入结构化配对回归及长上下文检查，再决定WY验收和正式性能；不靠继续换CANN或放宽门槛消除业务错误。

结果：results/cann921-upgrade-20261104/a3b-quality-pair921-v2/adjudication.json。新增无权重配套对照复现包results/upstream-gdn-cann921-pair-repro-20261104.tar.gz（10925B，脚本与实测冻结SHA一致、归档逐文件校验通过），可为现有PR17640补充测试；仍未公开发布。
