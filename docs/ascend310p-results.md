# Ascend 310P 优化结果

截至2026-11-04，生产参照为Qwen3.6-35B-A3B W8A8 TP2。Graph扩容未选中，WY-only有Prefill收益但未验收；新版配套H/O通过已测算子门禁，整模型业务仍未接受。27B已能在独立栈启动，尚无通过质量验收的性能基线。本轮无新增可上线改动。

## 范围与结果

a2隔离副本与a1生产使用相同镜像、W8权重和参数，驱动分别为26.1.1与26.0.rc1。W4预算支线与27B Dense分别建基线；Prefix Cache生产已启用，只控制命中，不计新增成果。

| 项目 | 本轮结果 | 边界 |
|---|---|---|
| W8 C1四档 | 1K/4K/16K/32K×256，12/12有效 | 每档3次，仅初筛 |
| W8并发 | C1/3/4各3轮，24/24成功 | 1K/256，非线上P95 |
| 完整业务形状回放 | 26/26，508.4s，76631输入/20690输出，缓存命中增量0 | 合成token输入及固定输出长度，用于计时，非纪要质量评测 |
| GSM64 | 63/64，1条512-token截断且缺最终答案 | 固定zero-shot子集，非全量标准分数 |
| 业务自动检查 | 完成/工具/schema 28/28；规范化事实16/16 | 不代表纪要语义全通过 |
| W8 Profile | 4K Prefill、32K首输出后Decode，2范围×2rank CSV均成功 | 诊断累计kernel耗时，不混入基准吞吐 |
| 工具回归 | 14项通过，Python编译及diff空白检查通过 | 测试范围是新增/修改的测量逻辑 |

配置身份与测量协议见[基线](ascend310p-baseline.md)，算子与捕获范围见[Profiling](ascend310p-profiling.md)。原始数据、恢复前后状态和说明位于results/production-w8a8-20261103/。

## 隔离基线与历史参照

### W8A8 TP2生产配置副本

权重10分片、93951张量、39,810,885,759字节，index/边界检查通过；a2全文件SHA、a1元数据SHA及长度一致，未在a1重读全部权重复核。每档3次、256输出、seed0、独立salt和零prefix命中。

| 输入 | Prefill中位tok/s | Decode中位tok/s | TTFT中位s | E2E中位s | 成功 |
|---|---:|---:|---:|---:|---:|
| 1024 | 973.65 | 32.90 | 1.054 | 8.804 | 3/3 |
| 4096 | 1290.43 | 32.44 | 3.178 | 11.028 | 3/3 |
| 16384 | 1204.27 | 30.86 | 13.615 | 21.871 | 3/3 |
| 32768 | 1082.54 | 29.12 | 30.286 | 39.037 | 3/3 |

C1/C3/C4聚合输出中位28.97 / 48.86 / 63.04 tok/s，分别3/3、9/9、12/12成功；该吞吐包含Prefill。冷缓存业务回放26/26、76631输入/20690输出、508.4s、40.7 tok/s，仅一轮合成形状计时。GSM64=63/64，index205在512 tokens截断、缺最终标记；parse_missing=1、truncated=1。事实门禁先14/16，两个误报均为多出“负责”前缀；仅去除一个前缀后16/16，字段仍严格，新增3项回归。自动业务28/28不覆盖纪要语义错误。

常规32K C1设备内存采样峰值device0为39138MB/44277MB、device1为38435MB/43693MB，共27阶段样本、间隔约5秒加查询开销；属于npu-smi采样值。证据：[W8基线摘要](../results/production-w8a8-20261103/README.md)。

### W4A8 TP1预算支线

镜像168个文件哈希及基础源码提交通过校验；52权重分片SHA256通过，93125张量中92160为expert张量。C16/MTP2/114688上下文/batch2048/seqs1/prefix off，质量前后8/8及A/B/A通过，固定seed310的GSM64=64/64、无解析缺失或截断。

| 输入 | Prefill中位tok/s | Decode中位tok/s | TTFT中位s | mean TPOT中位ms | 成功 |
|---|---:|---:|---:|---:|---:|
| 1024 | 590.35 | 34.93 | 1.738 | 28.627 | 3/3 |
| 4096 | 630.80 | 33.71 | 6.497 | 29.661 | 3/3 |
| 16384 | 594.65 | 31.96 | 27.562 | 31.269 | 3/3 |
| 32768 | 543.88 | 28.87 | 60.265 | 34.598 | 3/3 |

32K请求Prefill约60秒、Decode约9秒，仅说明该形状的E2E由Prefill主导。设备0内存采样最大39153MB、AICore100%，最大采样间隔5.64秒。W4后续Prefix候选在设备初始化退出139，未进入缓存推理；扩展矩阵取消，不能判定缓存不支持。W4结果独立于生产W8比较，原始制品编号`baseline-20261103`、`prefix-c16-20261103`。

### 历史参照

| 历史实验 | 结果 | 边界 |
|---|---|---|
| W4 MTP2 vs MTP1，112K Graph、32K/256 | Decode23.20→28.94 tok/s | 特定历史单流协议，非今日基线 |
| MindIEQ vs native W4 | Prefill8K +23.29%、32K +20.69% | 历史交替五次中位数，非新并发验收 |
| 10月31日W8 TP2 Graph MTP0 | 长会议509s、两场943s、短课堂209s | 业务阶段回放墙钟，不是单流Decode |
| 同轮W8 eager MTP2 | 478s / 838s / 212s | 出现职责错配，不能推荐 |
| GSM64 E5/D4 | 63/64，数字答案无变化 | 小子集，含共同截断；非完整精度等价 |

来源：[Qwen3.6](qwen36/README.md)、[MindIEQ](qwen36/mindieq-evaluation-20261014.md)、[业务回放](qwen36/business-replay-20261031.md)、[质量](qwen36/quality.md)。

## 第一项P0：Graph[1,2] → [1,4]

保持两张捕获图、TP2/W8/MTP0/262K上下文/batch4096/seqs4/prefix on等参数不变，验证三、四并发Decode覆盖。重试仅增加PYTHONFAULTHANDLER崩溃栈诊断。下表为1K输入、256输出、三轮的**含Prefill聚合输出吞吐中位数**。

| 并发 | 生产配置副本[1,2] tok/s | 候选[1,4] tok/s | 变化 |
|---|---:|---:|---:|
| 1 | 28.974 | 29.026 | +0.18% |
| 3 | 48.859 | 50.305 | +2.96% |
| 4 | 63.037 | 66.459 | +5.43% |

候选24/24计时请求成功，前后toy8/8、A/B/A一致性与16/16业务事实检查通过；两张图成功捕获，报告约0.09GiB。三并发未达到预先设定的至少5%门槛，因此控制器决定`not_selected`，未继续长回放、纪要语义或GSM复验。四并发结果只是三轮筛选，不能称已验收的业务提速或可靠P95。资源采样的候选较小读数也不能称节省内存：两配置之前的请求长度与allocator保留历史不同。

首次启动在加载模型及Graph捕获之前发生CPU FP16 zeros/fill段错误，未发推理请求；完整失败制品保存results/production-graph14-20261103/。同配置重试未复现该崩溃并完成筛选，数据保存results/production-graph14-retry-20261103/。目前不能归因为Graph事件资源不足，也不能由一次重试证明长期稳定。实验服务均已停止，NPU无运行进程。

## 算子证据与后续方向

W8 Decode主要关注MoE grouped matmul（约23%累计kernel耗时）、FP16 MatMul（主要变体约20%，lm_head另约9%）和PA mask（约12.5%）；recurrent GDN约2%，状态已FP16。Prefill有大量与WY Torch分解相符的Tril/Slice/Copy/ReduceSum。通用kernel不能全部归为GDN；Graph内MatMul shape缺失、HCCL完整任务和关键路径仍需进一步归因。

## 质量与采集问题的保留记录

事实门禁最初14/16，两个失败都是任务多了“负责”前缀，其他字段匹配；保留原始输出和失败状态，只允许去除一个该前缀后重算16/16，负责人/日期/任务/预算/待确认状态仍严格，新增3项回归。该误报曾触发实验服务停止，恢复时仅补跑未完成的回放/GSM/Profile，没有重跑已有效普通性能。

20条纪要对照仍发现基线风险，包括把行动安排扩写“已完成”、行动表漏提取已有职责/时间，见business-semantic-review.json。这是辅助审阅而非独立人工验收；本配置MTP0，不能把这些风险归因于MTP。

torch-npu daemon拒绝在线解析Profile；无设备离线导出又被msprof驱动初始化拒绝。原始采集均保留，补齐驱动/设备元数据访问后，同镜像离线analyse导出4份CSV成功。没有因此重跑普通基准，也没有把导出失败当作模型推理失败。

W4预算分支已完成基线、GSM64（64/64）及三份C16 Profile，其数值和算子排名不转用为W8结论。W4后续Prefix候选在设备初始化阶段退出139，未到health或缓存推理；制品见results/prefix-c16-20261103/，不能判定缓存不支持，扩展矩阵已取消。


## WY 单变量实验（2026-11-03，未验收）

固定并校验 GruntovDima 的 PR17640 head18163f6b，使用生产镜像构建 WY-only OPP/独立 namespace，保留生产 FwdH/FwdO。10 项算子数值、逐位重复、回退检查通过。a2 同配置候选完成四档 C1、stream、事实、业务形状回放和 GSM64；TP2 W8A8、Graph[1,2]、MTP0、262144 上下文与 Prefix Cache 保持基线。a1 未修改。

下表为每档三个有效冷缓存请求的中位数；属于初轮 A/B，随后执行原配置 A′ 复测以检查时间漂移。

| 输入/输出 | 基线 Prefill tok/s | WY 候选 Prefill tok/s | Prefill 变化 | Decode 变化 |
|---|---:|---:|---:|---:|
| 1K/256 | 973.65 | 1347.74 | +38.42% | −0.93% |
| 4K/256 | 1290.43 | 1987.52 | +54.02% | −0.80% |
| 16K/256 | 1204.27 | 1758.98 | +46.06% | −0.49% |
| 32K/256 | 1082.54 | 1510.58 | +39.54% | −0.72% |

1K/256 stream 聚合吞吐中位数：C1 29.737、C3 52.942、C4 67.215 tok/s，分别 +2.63%、+8.36%、+6.63%。完整冷缓存业务形状回放 26/26，20690 输出 tokens，492.81s，对照 508.40s，耗时减少约3.1%。Prefill 收益没有同比转化为长输出业务的总收益。启动 KV 预算7.74GiB/799035 tokens，与基线初次启动相同；未用缩短上下文换取吞吐。

候选 GSM64 为63/64、1条截断/缺最终标记，汇总与原基线相同；事实16/16和自动业务28/28通过。**候选尚未验收：会议4出现严重负责人错配**。源文按“分别为”指定周明负责当天客户回复、刘洋负责周三后端修复、何丽负责测试用例；候选表格轮换为周明修复、刘洋测试、何丽回复，并遗漏明确期限。原始基线该条映射正确，但随后原配置 A′ 在相同三并发整套测试中也出现同样的轮换错配；两配置重启后串行各6次均映射正确。因此不能把首次错误归为候选引入的回归，仍需检查并发序列及原路径风险。自动28条仅检查完成/工具/schema，未覆盖这个语义错误；控制器的 promising 状态不能替代语义验收。原始输出、辅助审阅和后续配对复测均保留，尚不能凭单次生成认定根因。

另发现原 Torch WY 路径在111-token synthetic输入上的偶发偏差，初测12轮原路径7/12在固定门限内、候选12/12且逐位稳定；独占复核原路径3/12、候选12/12，偏差仍集中在token56且最终状态逐位稳定，尚未定位WY/FwdH/FwdO边界。已在同一份失败输入上对照生产 CPU chunk 与独立逐 token 递推，输出最大绝对差3.59e−6、状态5.22e−8，支持参考的一致性，未定位 NPU 偏差根因。候选始终执行独立参考严格门限，未为追随原路径异常放宽标准。

四档 A′ Prefill相对最初 A 为−0.88%/+0.15%/+0.34%/+0.36%，Decode变化均在约±0.25%内；候选相对A′ Prefill仍+39.65%/+53.79%/+45.56%/+39.03%，支持收益未由跨时段漂移解释。A′长回放508.42s，与A的508.40s一致，B为492.81s。候选三并发整套业务3轮与独立双rank Profile均完成。会议4映射1轮错/2轮对，仍有期限漏提取。数值拆分证明WY与FwdH接口逐位稳定时，FwdO原始输出仍会偏离；候选冻结WY重放也有超限，不能称WY修复了数值问题。4K trace原生ComputeWy实际执行，kernel38072→8492，Decode仍380000。下一步独立验证FwdO vector-mask假设，尚未接受部署。详见 results/production-native-wy-20261103/README.md。


### FwdO 最小补丁排除与 Dense 准备（本轮续）

原始算子链中，WY/FwdH/FwdO 输入逐位稳定时仍可出现 raw FwdO 输出误差；这只能定位接口范围，尚未证明根因。原始 FwdO 单独固定输入重复调用可以通过，因此不能用该单项通过代替完整链门禁。

两个独立 API 的最小补丁均失败：向量 mask 改写、保留原算法仅加 V/S fence，各在 48 次固定输入验证中为 0/48，而配对原始算子为 48/48。没有启动补丁模型，没有修改生产 OPP；补丁不具备合入资格。构建框架差异仍需通过“不改算法、仅独立重编译”对照排除。对照准备过程的符号链接、安装目录权限及历史日志/容器名称冲突均保留失败记录；准备失败不计为数值结果。

核对镜像真实旧路径 csrc/moe/chunk_fwd_o、chunk_gated_delta_rule_fwd_h 和 common，59 个文件与生产提交 Git blob 一致。此前误查带 _vllm 后缀路径造成“源码缺失”的判断已撤销。

Qwen3.8-27B Dense 采用用户提供的 Eco-Tech/Qwen3.8-27B-w8a8-310p：26 文件、36,447,005,796 字节、9 个权重分片；逐文件固定 revision/SHA256、safetensors header/index 校验通过。完成下载不代表模型启动或推理通过。


### 原始重编译对照已完成；Dense 首次启动失败

原始 FwdO 算法独立重编译完成：固定输入 production/rebuild 各 48/48 在门限内且逐位稳定；原始完整链仍有 18/24 次越界（Torch WY 调用历史），本轮 frozen-WY-clone 为 0/24。故两补丁的 0/48 不可简单归咎于构建框架，而完整链异常尚未解决。证据 `results/fwd-o-rebuild-control-20261103/`。

Dense 完整生产配置启动已实际尝试：两个 rank 权重加载成功，各 19.2416 GB，随后首次 profiling/warmup 报 507015，设备日志指向 `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21`。没有请求、性能或质量结果，不能记为 Dense 基线完成。第一次同步诊断因 ACL Graph 不允许 ASCEND_LAUNCH_BLOCKING=1 而被参数校验拒绝；改 eager/Graph NONE 后，设备异常仍复现，Python 同步栈定位 MLP down_proj 的 310P W8A8_DYNAMIC Linear。待合成矩阵形状/布局诊断，未证明具体根因。

上游反馈草稿 `docs/upstream/310p-chunk-gdn-repeatability.md` 已整理，不依赖权重的最小脚本可复现原路径；尚未发布，也没有宣称修复。


Dense 合成量化矩阵隔离8项已完成：M=1/4096、两种MLP形状下，生产权重布局4项通过CPU采样整数积检查；另一种NZ布局2项数值错误，ND布局2项API拒绝。随机合成矩阵尚未复现真实模型 down_proj 的507015，因此不改生产布局；排队捕获实际输入。证据 `results/dense-quant-isolation-20261103/`。


### 现有上游 FwdO 单独复用门禁

GruntovDima PR17640 固定 head18163f6b，算法不改、仅提取独立 FwdO 绑定，生产镜像构建成功。保留生产 FwdH 的组合，固定输入上游 FwdO 0/48 在容差内（48/48逐位稳定），配对生产48/48通过；maxabs约0.004327、relative RMSE约0.195875，异常token0～6、64～70。严格门禁停止，未运行模型、未测性能，该项只测试混合组合。不能外推为上游完整PR错误。证据 results/upstream-fwd-o-20261103/。


### Dense 实际输入捕获与缓存对照

同步 eager 捕获仍复现启动异常，首次 down_proj 为M4096/K8704/N5120：INT8激活ND、INT8权重NZ转置view、F32通道/逐token scale；激活及逐token scale全零且有限。真实数据独立重放6项全部通过CPU采样整数积检查（maxabs0），含4096行原/ND/NZ激活布局和1/4/64行子集。零scale不是已证实根因；独立重建改变分配、描述和调用历史。首次重放在weights_only序列化Format枚举处失败，未运行算子；v2显式allowlist安装库Format类后验证通过，原记录均保留。

GDN执行器缓存容量对照：默认21/48、ACLNN_CACHE_LIMIT=0为25/48在容差内；完整Torch-WY链两边均12/24越界。未消除原异常，不能算修复。控制器清理时恢复a2临时unit触发systemd启动频率限制，原status保留failed；reset-failed后交接Dense独立缓存启动screen，最后统一恢复。

Dense问题草稿见docs/upstream/310p-dense-w8a8-startup.md；无模型权重的GDN复现包results/upstream-gdn-repro-20261103.tar.gz及SHA清单已生成，均未发布。


### 本轮最终状态

Dense在原Graph配置下仅设ACLNN_CACHE_LIMIT=0，仍启动失败507015，故障kernel仍为QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21；没有HTTP请求或性能数据。记录results/qwen38-dense-no-executor-cache-screen-20261103/。至此，生产布局随机矩阵4项通过、真实输入独立重放6项通过，但模型启动故障未定位，不能提交猜测性数值修复。

## 2026-11-04：Dense 权重存储与双设备交叉诊断

生产固定镜像下，模型内仅将down_proj权重重新分配并按原始物理[N,K] ND→NZ→transpose恢复，CPU逐元素严格相等检查通过。两个rank仍各加载19.2416GB；rank0首调用完成，rank1首调用仍507015、故障kernel/hash不变。无健康端点、无HTTP质量/性能结果，候选排除。完整实际worker与日志见results/qwen38-dense-quant-repack-20261104/。

核对发现此前实际输入6/6重放仅覆盖rank0/device0。补充两个TP捕获分片×两个物理设备交叉，4096行/原始布局，四项均PASS，96元素CPU采样最大误差0且全输出finite。输入与pertoken_scale全零，不能作为一般非零精度结论。单独分片或设备不足以复现，完整调用历史/分配状态仍待定位。记录results/dense-quant-rank-device-grid-20261104/。

另审计两套已有post4镜像，只读元数据归档results/dense-runtime-audit-20261104/。CANN安装文件均自报9.1.0/内部V100R001C11SPC001B243，历史9.1.1标签不能替代本次读数。选较接近的vLLM0.23镜像作完整运行栈兼容性筛选，不改生产镜像或驱动；不能归因为单独torch-npu升级。首次入口脚本差异导致读取/model失败，尚未加载权重，单独保留；显式入口重试另记。

### 真实输入独立进程间歇507015已复现

显式expandable=True四格3PASS/1FAIL（rank1/device0，原kernel/hash）。继而rank1/device0 False/True交替5次：False4PASS/1FAIL507015；True3PASS/1FAIL507015，最后1项初始化HDC等待128.88s后停止，未到算子，单独记为未完成。原status=failed保留。关闭expandable不能作为修复，早期独立重放通过不代表该故障只能在模型内出现。记录results/dense-quant-expandable-grid-20261104、dense-quant-allocator-paired-20261104。

post4 v3已通过profile_run，随后KV准入失败（8.09GiB需求/7.66GiB可用）；与生产dynamic W8 Python代码逐字节相同，但CANN/torch-npu及Ascend提交均不同，不能定位到单独组件，也不能凭一次通过宣称间歇错误已修复。vLLM提交仍0fc695fc，Ascend为1cdb8c4d。当前保持262144上下文，在同镜像上以利用率0.94单独检查容量，不通过缩短上下文制造可运行结论。

### Dense post4：完整上下文准入成功，严格短题7/8未验收

同post4同步eager，仅利用率从0.90调整0.94，保持maxlen262144/batch4096/seqs4/prefix on/MTP off。缓存预算rank0/1约8.50/8.63GiB，报告KV274795tokens/满长请求并发1.05×；缓存初始化、health200、HTTP推理已实际到达。这里未实际发送256K长请求，不代表多路满长并发。

第一轮quality-before严格7/8：排序题期望`2,7,9`，实际`279`（漏逗号），原始输出和failed状态保留；A/B/A干扰前后输出一致、canary未泄漏。quality-after/GSM/业务和性能均未执行，不放宽门槛、不作为通过基线或上线推荐。记录results/qwen38-dense-post4-runtime-kv94-20261104。


## 完整上游 H/O 组合通过算子门禁（2026-11-04）

固定PR17640 head18163f6b，在生产post2镜像构建成对FwdH/FwdO及保留NZ初始状态逻辑的原始binding，独立namespace/vendor。与此前新O/旧H混合组合分开记录。两张卡、Torch/原生WY四组完整链各24次加冻结克隆24次，共192/192在误差门限内，raw输出逐位稳定，最大绝对误差6.3153e-6。两个新进程各10/10严格门禁（每组6项完整GDN输出/最终状态，3项WY布局，1项CPU回退）通过，多形状输出最大误差8.2599e-6，状态最大8.8640e-5；Torch WY+成对H/O也通过CPU参考。原始probe和严格门禁有不同检查范围，不能把192次都说成重新运行WY或逐次state验证。

这是对既有上游实现的生产栈验证，不是新原创kernel；完整组合尚未完成模型/业务/Logits等价与性能验收。模型判定见下一节。首次准备失败和成功重试分别见results/upstream-gdn-pair-v{1,2}-20261104；贡献范围审计见results/upstream-gdn-pr-audit-20261104。

## 2026-11-04：完整H/O候选本轮最终判定

配套H/O+原生WY算子验证通过（四组192次输出检查、两次严格10例），模型质量筛选已结束。前后短请求各8/8，事实16/16，业务自动28/28；人工会议4仍有负责人错配及完成状态编造，因此最终not_accepted_business_semantics，不进入性能或部署。GSM64=63/64，与冻结生产基线逐样本答案/正确性一致，唯一index205截断；文本19/64完全一致，不证明逐token等价。A′此前也有业务错误，不能据此将错误归因为候选独有回归。

结果：results/production-gdn-pair-quality-20261104/adjudication.json。上游路线和补充研究取舍：docs/research/gdn-upstream-pr-evaluation.md。

## 2026-11-04：CANN9.2.1完整默认栈隔离验证

用户同意后在a2拉取官方v0.30.0-310p（IDd103c59b31c6…），核实Toolkit/OPP/NNAL9.2.1、torch-npu2.10post7、vLLM/Ascend0.30.0；a1和宿主驱动固件未改。初始GDN T111两卡96/96、Dense无权重Matmul10/10通过，但扩展门禁13/15，两组T512输出偏差。独立进程T512 HV16仅29/48在容差内、HV24为0/48，末state各48/48稳定。最终not_accepted_gdn_correctness；未加载模型、未测性能，不宣称CANN单独修复Dense或GDN。

完整评估docs/research/cann-upgrade-evaluation.md；原始results/cann921-upgrade-20261104；可公开合成复现results/upstream-gdn-cann921-repro-20261104.tar.gz（尚未发布）。本轮至此保留隔离新版环境，下一步优先GDN输出正确性，不扩大WY重复实现。

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


## 2026-11-04：字段与长上下文配对回归完成

同a2串行测试原生产A与9.2.1+PR17640 paired H/O候选B，两者均35B-A3B W8A8 TP2、原Torch WY。13种独立预期用例×C1/C3，每栈26/26通过，共52请求；其中4095/4096/4097/16384/32768合成输入每栈10/10。两个栈13种实际token ID哈希一致，所有请求usage计数匹配；四次前后短题检查均8/8。仅行动项顺序可忽略，其他字段严格一致。

原会议4/5/18另做9个同payload请求/栈并人工审阅。会议4负责人映射A为2/3、B为3/3，样本量不足以证明修复；双方均3/3将当天回复补成已完成。会议5双方均3/3补成已执行停用；B的C1再次把当天改为本周内（已执行），A三次均遗漏该行动项行，不能因A未写错期限就说A完整正确。B的同一行还将所有部门细化为各部门负责人。会议18双方全文保留角色/时间关系，但行动项结构用未明确，按结构损失记录。完成状态采用事先冻结的保守标准，原短句本身存在时态歧义，不将该人工判定作为数值误差证据。

最终structured_gate=passed_for_tested_inputs，整体仍not_accepted_business_semantics。保留截止时间差异作为下一步受控复核样本；不宣称CANN/H/O单独导致差异。未进行新吞吐/GSM/logits验收，不把合成32K提取当作真实长会议或262K质量。此前38–54%收益不得迁移到新控制组。原生WY尚未在9.2.1验收，必须重新编译并通过算子及相同模型门禁；正式性能仍等待质量过关。Decode/Dense继续位于正确性工作之后，不扩展重复WY PR。

结果：results/action-state-paired-v1-20261104/，以adjudication.json为最终判定。a2原服务active且独立18081/health=200，无运行rootless实验容器，驱动仍26.1.1；未操作a1、宿主安装或公开发布。冻结工具/fixture SHA和新增三脚本编译校验通过。


## 2026-11-04：27B新版16K实测及排序/标点诊断

27B W8A8 TP2在固定9.2.1+PR17640配套H/O+原Torch WY下实际健康启动，maxlen16384，Graph[1,2]、MTP0、prefix on。两rank日志各19.2416GB权重，实际映射paired API；本轮未复现旧507015，但不宣称所有形状/长上下文均修复。

首轮短题7/8，排序期望2,7,9而输出279，和旧post4同题输出相同。ABA一致、无canary泄漏。按预定门禁停止事实/业务/性能；20条语义准则提前冻结但未执行，不能报告27B会议质量或与35B的业务排名。

首轮原服务独立HTTP200后再启动窄范围第二轮，保持同配置。6类×3重复：原排序3/3输出279，raw token IDs=[17,22,24,248046]，未生成逗号11；数字逗号复制、字母逗号复制、分号、数组复制和JSON schema排序五类15/15通过。原排序首个样本生成2后，下个token7的logprob约-0.257、逗号约-1.491，greedy选7。CPU tokenizer亦可完整往返2,7,9。该例定位到提示词相关的生成选择，不是已观察到的API解析删逗号；仍不能区分基础模型与量化影响。JSON schema改变约束，其通过不能覆盖原题失败。

两次均not_accepted_short_format_gate，未执行正式性能或业务比较。证据位于results/cann921-upgrade-20261104/dense-quality-pair921-16k-v1及dense-quality-format-pair921-16k-v1。新增短上下文容量选项、双worker库验证、独立诊断模式与原始token/logprobs留存。下一步保留失败，围绕同模型参考/精度与原始提示词对照定位，不为标点错误盲改GDN算子，也不扩大八种profile的可用声明。


### 窄范围35B对照与下一步

原服务恢复HTTP200后，a2原35B W8A8部署完成同样6类×3次请求，18/18符合预期；27B为15/18，差异仅原排序的3次重复（35B3/3、27B0/3）。请求除model字段外严格一致；完整18组配对见dense-quality-format-pair921-16k-v1/a3b-narrow-comparison.json。运行栈、模型/权重、上下文配置不同，不能将其称为单独模型能力或量化因果对照，亦不是会议业务评测。

原短题门禁保持7/8未通过；本轮没有绕过门禁跑业务/性能。下一阶段将格式遵循反例与算子数值验收分开，允许另立明确“不予验收”的业务诊断轮收集27B原会议/结构化结果，不能用诊断完成替代质量门禁通过。已有token证据不支持为此反例继续盲换CANN或改GDN；若要归因基础模型与量化，仍需同模型参考。正式性能和可部署profile均等待验收。

收尾：原服务active、18081/health=HTTP200，无运行rootless实验容器；驱动仍26.1.1，a1与宿主安装未变。两轮冻结工具SHA校验、第二轮源码与本地一致、两脚本编译和diff/新增源空白检查通过。上游未发布。


### 2026-11-04：本机GPU可复用参考已找到

本机GPU_HOST有3090 24GiB（可用约0.83GiB）和3080本机报告20GiB（可用约6.98GiB），均已有服务，采样瞬时0%不代表显存空闲。3090现有18020为Qwen3.8-27B-W4A16-AutoRound-fast，FP8 KV+MTP3，健康200。未停服务或另起模型，复用其完成同18条窄诊断：GPU18/18、310P15/18，原排序GPU3/3为2,7,9、NPU3/3为279。请求除model字段完全相同。

GPU/NPU tokenizer和模板文件SHA不同，但原排序实际chat输入37个token ID逐项相同（GPU/tokenize与NPU原制品无设备离线AutoTokenizer核对）。该例进一步定位至生成侧部署差异，不能据此将279归为27B固有能力不足或API解析问题；量化、KV、MTP、运行栈和底模revision仍为混杂因素。GPU不是高精度参考，W4通过不证明优于W8。下一步可先用现有端点补业务语义诊断，不为对照打断本机服务；严格归因另需统一条件。结果results/local-gpu-reference-20261104/。
