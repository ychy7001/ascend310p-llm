# CANN完整栈评估与采用条件

2026-11-04冻结评估。新版完整栈用于隔离诊断，生产采用判定见[结果](../ascend310p-results.md)。

## 固定版本与兼容依据

已通过官方GitHub API和固定tag原始文件查询并归档至 `results/cann-upgrade-audit-20261104/`：

- vLLM-Ascend v0.30.0于2026-10-29 14:07:04 UTC发布，API标记prerelease=false。
- 固定v0.30.0兼容矩阵为：vLLM/vLLM-Ascend均v0.30.0、CANN9.2.1、PyTorch2.10.0、torch-npu2.10.0.post7；不能随意混合版本。
- 固定v0.30.0的Dockerfile.310p使用 `quay.io/ascend/cann:9.2.1-310p-ubuntu22.04-py3.12`，构建默认vLLM_TAG=v0.30.0；目标存在明确310P构建路径。需固定最终镜像digest，不能凭tag认定本地已有同一版本。
- 查询时main安装文档列出Ascend HDK26.1.1、NNAL9.2.1及Toolkit+Ops9.2.1，明确Atlas300IDUO对应310p镜像。a2驱动version.info为26.1.1，与该驱动版本相符；**尚未独立核实固件版本及发行要求全部匹配**，不把此项写成完整HDK通过。
- 发行说明包含Qwen3.8-27B、混合线性注意力缓存、GDN prefill及310P量化部署相关修复。不能从这些概括推断本轮QuantBatchMatmulV3_NZ错误hash或共享FwdO偏差已修复，需分别重放合成复现及真实模型。
- 旧栈身份见[基线](../ascend310p-baseline.md)，post4完整栈比较见[结果](../ascend310p-results.md)。

额外兼容点：v0.30.0的310P Dockerfile显式卸载triton-ascend/triton。通用兼容矩阵的Triton版本不能当作310P开箱支持的证明；#17604 Triton prepare依然只作算法参考。CANN9.2.1基础镜像也不自动证明包含#18057所需的外部ops-transformer固定commit及其Python API。

## 实测环境

官方v0.30.0-310p镜像ID `d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a`，实际Toolkit/OPP/NNAL9.2.1、torch-npu2.10post7、vLLM0.30.0+empty、Ascend0.30.0。a2物理卡ID8、两个逻辑310P3，驱动26.1.1、固件7.5.0.5.220，设备报告Compatibility=OK；宿主未覆盖安装，固件与发行要求的完整对应仍未独立核实。

## 默认栈与配套H/O

初始T111两卡96/96通过，Dense合成seed310..314×allocatorFalse/True共10/10通过且本轮无507015；扩展默认GDN15组仅13/15通过。独立新进程T512/seed318复核，35B对应HV16为29/48在容差内，Dense对应HV24为0/48，末state两组48/48在容差且逐位稳定。最大输出abs约0.00504/0.00599，相对RMSE约0.0652/0.0749，原阈值仍为atol2e-4、rtol0.04、relativeRMSE<=0.02。不能把小形状通过概括为GDN修复。

使用上述固定镜像及同一工具链重新编译PR17640固定head18163f6b的成对H/O，独立vendor/namespace，保留原始NZ状态binding。**Torch WY保持默认**，没有加载旧9.1 WY二进制或改算法。

| 检查 | 新版默认栈 | 9.2.1配套H/O、Torch WY不变 |
|---|---|---|
| T512/seed318/HV16，device0 | 29/48在容差内，输出1/48等于首轮 | 48/48在容差内且输出/末state逐位稳定 |
| T512/seed318/HV24，device0 | 0/48在容差内，输出1/48等于首轮 | 48/48在容差内且输出/末state逐位稳定 |
| 同输入device1 | 本轮默认对照未覆盖 | HV16、HV24各48/48通过 |
| 15组严格边界 | device0为13/15 | 两卡各15/15，每组额外3次逐位重复 |

配套H/O的T512输出maxabs为6.54e−6/6.38e−6，relativeRMSE约0.000424；15组输出最大maxabs1.025e−5、末state1.2465e−4，原门限atol2e−4、rtol0.04、relativeRMSE<=0.02未改变。结果说明此配套组合能消除已测样本异常，尚未区分H或O根因，不是整模型质量/性能验收。

验证前纠正加载顺序：vLLM bootstrap会将默认vendor置前，而9.2.1默认库也导出相同Vllm API名；独立NPUBridge会在dlopen时捕获环境。故在bootstrap之后、dlopen之前重新前置paired vendor，并记录API/扩展SHA和实际映射库。算法/编译源未改。6个验证进程均只映射配套libcust_opapi.so。原始通用probe头仍称default/original scope，实际实验范围以wrapper日志和adjudication为准，未修改原始jsonl。

证据：[upstream-gdn-pair921-v1-20261104](../../results/upstream-gdn-pair921-v1-20261104/README.md)。模型筛选与最终判定见[结果](../ascend310p-results.md)，逐字段复核见[配对研究](action-state-paired-validation.md)。

## 采用条件

1. 固定正式发行的完整兼容组合及digest；核对真实Toolkit、OPP、NNAL、torch-npu、vLLM和Ascend身份。构建清除旧产物，独立缓存/vendor，验证实际映射库。
2. 先保留默认路径的独立GDN与Dense合成复现，再单变量验证配套修复。默认路径失败和配套组合通过分别留证；完整栈效果不归因单独CANN。
3. 整模型先验短题、状态复用、事实/关系、同索引GSM、长上下文/chunk边界，按需要补固定token或logits检查。质量与数值验收分开；Dense须建立独立可复现基线。
4. 对应模型数值门禁通过后才测正式Prefill/Decode、业务形状墙钟、内存与启动代价，沿用相同TP、长度、输出、预热及零命中协议。生产采用另外要求业务通过。

原生WY重建的算子覆盖见[Profiling](../ascend310p-profiling.md)，后续顺序统一见[计划](../ascend310p-optimization-plan.md)。本地无权重复现包编号`upstream-gdn-cann921-repro-20261104.tar.gz`、`upstream-gdn-cann921-pair-repro-20261104.tar.gz`，后者10925B且逐文件SHA通过，均尚未公开发布。
