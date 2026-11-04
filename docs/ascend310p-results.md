# Ascend 310P 优化结果

截至2026-11-04，生产参照为Qwen3.6-35B-A3B W8A8 TP2。Graph扩容未选中，WY-only有Prefill收益但未验收；新版配套H/O通过已测算子门禁，整模型业务仍未接受。27B已能在独立栈启动，尚无通过质量验收的性能基线。本轮无新增可上线改动。

a2隔离副本与a1生产使用相同镜像、W8权重和参数，驱动分别为26.1.1与26.0.rc1。W4预算支线与27B Dense分别建基线；Prefix Cache生产已启用，只控制命中，不计新增成果。配置身份与协议见[基线](ascend310p-baseline.md)，算子证据见[Profiling](ascend310p-profiling.md)。

## 隔离基线与历史参照

### W8A8 TP2生产配置副本

权重10分片、93951张量、39,810,885,759字节，index/边界检查通过；a2全文件SHA、a1元数据SHA及长度一致，未在a1重读全部权重复核。四档共12/12有效；每档3次、256输出、seed0、独立salt和零prefix命中。

| 输入 | Prefill中位tok/s | Decode中位tok/s | TTFT中位s | E2E中位s | 成功 |
|---|---:|---:|---:|---:|---:|
| 1024 | 973.65 | 32.90 | 1.054 | 8.804 | 3/3 |
| 4096 | 1290.43 | 32.44 | 3.178 | 11.028 | 3/3 |
| 16384 | 1204.27 | 30.86 | 13.615 | 21.871 | 3/3 |
| 32768 | 1082.54 | 29.12 | 30.286 | 39.037 | 3/3 |

C1/C3/C4聚合输出中位28.97 / 48.86 / 63.04 tok/s，分别3/3、9/9、12/12、共24/24成功；该1K/256吞吐包含Prefill。冷缓存业务回放26/26、76631输入/20690输出、508.4s、40.7 tok/s，仅一轮合成形状计时。GSM64=63/64，index205在512 tokens截断、缺最终标记；parse_missing=1、truncated=1。事实门禁先14/16，两个误报均为多出“负责”前缀；仅去除一个前缀后16/16，字段仍严格，新增3项回归。自动业务28/28不覆盖纪要语义错误。

常规32K C1设备内存采样峰值device0为39138MB/44277MB、device1为38435MB/43693MB，共27阶段样本、间隔约5秒加查询开销；属于npu-smi采样值。证据：[W8基线摘要](../results/production-w8a8-20261103/README.md)。

### W4A8 TP1预算支线

镜像168个文件哈希及基础源码提交通过校验；52权重分片SHA256通过，93125张量中92160为expert张量。镜像`b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d`；C16/MTP2/114688上下文/batch2048/seqs1/prefix off/利用率0.95，质量前后8/8及A/B/A通过，固定seed310的GSM64=64/64、无解析缺失或截断。

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

## Graph[1,2] → [1,4]

保持两张捕获图、TP2/W8/MTP0/262K上下文/batch4096/seqs4/prefix on等参数不变，验证三、四并发Decode覆盖。下表为1K输入、256输出、三轮的**含Prefill聚合输出吞吐中位数**。

| 并发 | 生产配置副本[1,2] tok/s | 候选[1,4] tok/s | 变化 |
|---|---:|---:|---:|
| 1 | 28.974 | 29.026 | +0.18% |
| 3 | 48.859 | 50.305 | +2.96% |
| 4 | 63.037 | 66.459 | +5.43% |

候选24/24计时请求成功，前后toy8/8、A/B/A与16/16事实检查通过；两图捕获约0.09GiB。C3未达预设5%门槛，最终`not_selected`，未继续长回放、纪要语义或GSM复验。C4仅三轮筛选；内存读数受先前请求和allocator保留历史影响，不作节省结论。

首次启动在模型加载和Graph捕获前CPU FP16 zeros/fill段错误；同配置重试未复现。该启动异常不能直接归因为Graph事件资源不足，失败和重试制品分别编号`production-graph14-20261103`与`production-graph14-retry-20261103`。

## 算子结论

Prefill的WY Torch分解值得替换，Decode优先关注MoE、FP16投影/输出头和PA mask。采集边界、双rank占比、WY kernel变化及GDN/Dense数值诊断统一见[Profiling](ascend310p-profiling.md)。

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

候选GSM64为63/64、1条截断/缺最终标记，与基线相同；事实16/16、自动业务28/28。会议4将周明的当天回复、刘洋的周三修复、何丽的测试用例轮换成周明修复/刘洋测试/何丽回复，并漏期限。原始A映射正确，A′相同三并发整套测试也出现轮换；重启后两配置串行各6次均正确。该错误超出自动业务检查覆盖，尚未隔离为候选独有回归。

A′相对最初A的四档Prefill为−0.88%/+0.15%/+0.34%/+0.36%，Decode均约±0.25%；B相对A′仍+39.65%/+53.79%/+45.56%/+39.03%。A′回放508.42s，与A的508.40s接近，支持收益未由时间漂移解释。候选三并发整套业务3轮，会议4映射1轮错/2轮对、仍漏期限；WY提速未解决完整链数值异常，保持未验收。

两个最小FwdO补丁、新O/旧H混合组合、关闭executor cache均未通过，原算法重编译对照未支持“仅构建框架不兼容”的解释。详细数值及复现见[Profiling](ascend310p-profiling.md)，上游契约见[研究](research/gdn-upstream-pr-evaluation.md)。

## Dense旧栈兼容与容量

Eco-Tech/Qwen3.8-27B-w8a8-310p共26文件、36,447,005,796字节、9权重分片，固定revision/SHA256及safetensors header/index通过。生产栈两rank各加载19.2416GB后报507015；同步eager和关闭executor cache均未解决，未获得请求性能基线。

原权重存储重建、设备交叉及allocator对照未得到可靠修复；真实输入和合成权重均可独立进程复现间歇507015，见[诊断证据](ascend310p-profiling.md)。

post4完整镜像实读CANN9.1.0/内部V100R001C11SPC001B243、torch-npu2.10.0.post4，vLLM仍0fc695fc，Ascend为1cdb8c4d；dynamic W8 Python源码与生产逐字节相同。该对照改变多个运行栈组件。

post4 v3通过profile_run后，在利用率0.90因KV需8.09GiB、可用7.66GiB拒绝准入。该单次通过不足以证明间歇故障已修复。

### Dense post4：完整上下文准入成功，严格短题7/8未验收

同post4同步eager，仅利用率从0.90调整0.94，保持maxlen262144/batch4096/seqs4/prefix on/MTP off。缓存预算rank0/1约8.50/8.63GiB，报告KV274795tokens/满长请求并发1.05×；缓存初始化、health200、HTTP推理已实际到达。这里未实际发送256K长请求，不代表多路满长并发。

第一轮quality-before严格7/8：排序期望`2,7,9`、实际`279`；A/B/A输出一致、无canary泄漏，quality-after/GSM/业务和性能未执行。格式判定统一见下文摘要。记录[qwen38-dense-post4-runtime-kv94-20261104](../results/qwen38-dense-post4-runtime-kv94-20261104/README.md)。

## 生产post2配套H/O＋原生WY

算子检查见[Profiling](ascend310p-profiling.md)。前后短请求各8/8，事实16/16，业务自动28/28；人工会议4仍有负责人错配及完成状态编造，因此最终not_accepted_business_semantics，不进入性能或部署。GSM64=63/64，与冻结生产基线逐样本答案/正确性一致，唯一index205截断；文本19/64完全一致，不证明逐token等价。A′此前也有业务错误，不能据此将错误归因为候选独有回归。

结果：`results/production-gdn-pair-quality-20261104/adjudication.json`。候选契约见[上游研究](research/gdn-upstream-pr-evaluation.md)。

## CANN9.2.1默认栈与配套H/O

默认完整栈扩大GDN门禁失败，最终`not_accepted_gdn_correctness`，未进入整模型。保留Torch WY、重新编译配套H/O后，已测GDN异常消除；它是可继续验证的隔离组合，尚不构成生产采用。版本身份、逐形状误差和采用条件统一见[CANN评估](research/cann-upgrade-evaluation.md)。

## 9.2.1配套H/O＋Torch WY的35B质量

该组合通过算子门禁后实际启动Qwen3.6-35B-A3B W8A8 TP2，保持maxlen262144、batch4096、seqs4、Graph[1,2]、SSM FP16、MTP0、prefix on，利用率0.90。各rank权重18.8545GB；完整上下文配置通过准入，不代表已发送262K请求。独立检查两个worker均映射paired API库，并记录原默认库仍服务其他算子。

前后短题各8/8、ABA输出一致无canary泄漏，事实16/16，业务自动28/28（20会议、3tool、5schema），但人工语义仍不通过。会议4本轮周明/刘洋/何丽映射正确，却把未标明完成状态的当天回复写成“已于当天完成回复”；会议5同样编造已完成停用，并把行动项截止时间从“当天”写成“本周内（已完成）”。原生产A也出现前两项完成状态编造，不能当作本候选独有的数值回归；本周内是本轮额外差异，仍需配对复核。20会议均已人工阅读。

28项原始请求与原生产A逐项严格相同；25条非tool文本7条相同，文字变化不能直接等同质量或数值退化。最终not_accepted_business_semantics，无新GSM、性能和生产采用。

结果：`results/cann921-upgrade-20261104/a3b-quality-pair921-v2/adjudication.json`。复现与贡献范围见[上游评估](research/gdn-upstream-pr-evaluation.md)。

## 字段与长上下文配对

结构化字段在已测输入通过，原自由文本的完成状态、期限和行动项结构仍有问题，整体采用沿用上节判定。冻结输入、双方分母、逐会议差异及覆盖边界统一见[配对验证](research/action-state-paired-validation.md)。

## 27B与GPU行为对照摘要

27B在post4完整上下文与9.2.1配套H/O的16K配置均因短题格式门禁失败，保持`not_accepted_short_format_gate`，无正式性能基线。排序token/logprob、GPU27B/GPU35B业务结果、short-4与证据协议复核、模型归因边界及部署候选统一见[模型质量研究](research/model-quality-and-deployment-profiles.md)。现有证据不支持将35B业务错误视为310P独有，也不足以按自动检查选择27B替代35B。
