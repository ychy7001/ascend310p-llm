# Ascend 310P Qwen3.6 低比特推理：技术反馈

2026-10-12，北京时间。硬件实验使用Atlas 300I DUO上的310P3；以下为研究结果，不是华为或Qwen官方认证实现。

## 可复现状态

| 技术 | 当前状态 | 验收边界 |
| --- | --- | --- |
| Routed Experts W4A8 | 完整模型TP1/TP2启动、生成 | 其它层保持浮点；不保证优于W8A8 |
| W4A8+MTP | Qwen3.6自身BF16 MTP，TP1/TP2生成与部分benchmark | 一枚草稿TP1/TP2；两枚草稿仅TP1/FP16 KV/eager/32K已测 |
| C8 | 独立page写入、读取、反量化、Attention与mutable Graph已测 | 软件C8持久存储、按需转FP16 NZ后原生PA；不是原生INT8 PA |
| W4A8+MTP+C8+Decode Graph | TP1/4K真实Graph replay与质量检查，接近128K单次检索通过 | 仅MTP=1；长输入重复质量、正式尾延迟、TP2 C8与32G实卡门禁未完成 |

环境：镜像`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，CANN9.1.0，Torch2.10.0+cpu，torch-npu2.10.0.post4，vLLM基于`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，vLLM-Ascend基于`1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。额外社区W4模块及本地修改见来源说明。

## 设备实测摘要

- W4 TP1权重加载约23.82GiB；无MTP配置262144/.95可启动，精确32768及131072 prompt实际生成通过。配置上限与KV容量估计不等于256K实际生成验收。
- W4+MTP TP2 p128/o256单流3轮约27.69–28.49tok/s；既有同W4无MTP约26.05tok/s，探索性平均约+7.9%。并发4约87.36–96.51tok/s。尚不足以推广为全部负载的收益或严格W8A8对比。
- W8/W4同形状profile：Decode专家累计kernel约32.50→22.77ms，Prefill约43.10→108.75ms。通信、非专家和布局成本抵消部分收益；这些累计时间含不同执行组织，不能归因为唯一位宽变量。
- C8 NZ两次转置、紧凑scale实现32K纯转换约924μs；相同tile32双缓冲约1122→979μs，仍未优于已有tile64配置。完整转换+原生PA仍慢于FP16，具体复现见C8说明。
- 32K、2KV、dim256时C8持久存储34MiB，FP16 K/V64MiB；C8 staged路线另需64MiB临时NZ。必须跨层复用并计入工作区，尚未证明完整模型内存净节省。

以上来自a2实验；原始日志留在实验机，仓库保留汇总、源码与复现命令。首版公开checkpoint的独立加载结果另见`release-validation.md`。

## 请求华为提供的具体指导

1. CANN9.1/Ascend310P3是否有可用C8 paged-attention及写cache实现？请明确接口、产品支持矩阵、NZ/ND布局、scale粒度、page stride和最小shape。目标full attention为16Q/2KV、dim256、block64。
2. half×int8高阶Matmul是否经UB antiquant？基础`Mmad<float,half,int8_t>`编译失败，half/half同shape控制组成功；SDK源码含反量化。是否有避免显式FP16 tile的可用路径？该编译限制不等于全部硬件潜在能力。
3. C8→NZ的双转置和事件控制成本应如何降低？UB→L1在该SoC有哪些支持/软件模拟边界？希望提供小M/GQA适用的tiling、流水和layout建议。
4. Routed Expert Gate/Up为K2048/N1024，Down为K512/N2048；per-token A8按INT4 limbs组合mad_s4。如何优化Decode小M与Prefill路由分组、L1权重复用、跨K累加？哪些优化不能从910B/A2/A3直接迁移？
5. Qwen3.6 full/linear attention混合cache、动态token scale、MTP状态回退应如何分组和物理对齐？当前官方C8开关在hybrid page unification阶段失败，不能仅绕过检查。

## 下一验收节点

C8已实现TP1/eager/单请求短生成，关闭MTP，仅压缩full-attention K/V，linear-attention状态维持原精度。匹配FP16 KV对照：p512/o64中位吞吐14.2881→13.2868tok/s（−7.01%），TPOT45.40→49.80ms；p3072/o32吞吐2.9583→2.8152（−4.84%），TPOT45.90→50.91ms。都是1warmup/3rounds、4K/.90/max-batch256，同一prompt IDs，非尾延迟统计或正式精度。五类sanity均通过，三类文本相同、两类文本不同，不证明精度等价。按层复用NZ scratch；验证chunked prefill、decode append、页复用、生成质量与实际峰值内存，再加MTP拒绝token/state rollback，最后TP2/并发/Graph/长上下文。

## 2026-10-12 下午长输入与Prefill补充（experimentally verified）

以下来自同一a2/chip0，TP1、FP16 KV、MTP=0、eager、max len32768、memory .90、max seq1、prefix off、scheduler batch4096/expert chunk640。固定token IDs，warmup1后三轮；Prefill与Decode为服务端请求阶段计时，非纯kernel时间。

| 输入/输出 | Prefill中位 tok/s | Decode中位 tok/s | HTTP TTFT中位 | 检查边界 |
| --- | --- | --- | --- | --- |
| 8192/1 | 384.2473 | 不适用 | 21.518s | 三轮counter与usage通过 |
| 32256/1 | 358.0136 | 不适用 | 90.365s | 三轮counter与usage通过 |
| 8192/256 | 384.8600 | 21.0647 | 21.3138s | 三轮持续生成；平均TPOT约47.48ms |
| 32256/256 | 358.1915 | 18.8091 | 见原始请求 | 三轮持续生成；非正式生成准确率 |

32256token中间needle另测通过，14输出token、94.0191s，正确返回验证码；单个检索样本不代表完整质量。8K→近32K，Prefill约−6.8%、Decode约−10.7%；不得将其他MTP/Graph/C8配置的约30tok/s与本表连成同配置曲线。相同32K条件下1024/512对照已完成，8K/近32K Prefill及Decode与4096/640差距不到1%，暂保留1024/512。源码与复现`experimental/run-long-context-sweep.sh`、`tools/bench_server_phases.py`；详细证据见`prefill.md`。

NPU profile：3072输入/output1，400次W4专家MatMul累计3.980758s，占op_statistic累计时间50.402%；Slice .422909s、Mul .361105s、ReduceSum .291646s、ViewCopy .279497s；不是普通请求墙钟占比，也尚无实测GB/s证据。采集入口`tools/profile_http_prefill.py`。希望协助判断激活limbs、矩阵tile、按专家分组与数据搬运的优化方向。

TP1/48G、W4+MTP1+动态C8+Decode Graph的另一轮`.97`服务曾完成130560输入+12输出中间检索，587.262s；这不是本表配置，且不构成稳定128K交互规格。此前`.95`近上限请求因hybrid scheduler容量门禁不足而等待，修正按所有group及reserved block估算后才完成。请关注hybrid状态页、speculative状态、Graph与共享scratch的联合内存规划。

补充请求：在上述明确配置下，Prefill/Decode随历史长度的变化是否合理？310P KV预算halved for workspace中的工作区能否安全复用/按需缩小？软件C8到FP16 NZ的scratch是否有受支持的融合替代？希望给出32G/48G产品预算指导，同时保留长Prefill与并发时余量；不依靠单纯提高memory utilization发布规格。

## MTP2与Graph转换优化补充

experimentally verified：同TP1/FP16KV/eager/32K/.90/1024:512/maxseq1/prefix off，MTP1→MTP2，8K输入/output256 Decode中位26.2373→33.1059tok/s，近32K 22.4596→28.8517tok/s；近32K Prefill352.0753→350.2415tok/s。两组各toy8/8、A/B/A无泄漏且文本一致、近32K单检索通过。固定重复文本长请求两位置接受率100%，非业务普遍收益；p128 MTP2草稿620/接受490，第二位置接受229，实际存在未接受草稿。完整质量与MTP2 Graph/C8仍未验收。

C8图内转换候选：新可选NPU INT32 context_length在kernel侧限制有效页，输出extent及Graph shape不变；独立Graph+原生PA测试七个长度/页表变化全部max_abs=0。固定128K extent、warmup5+30次交替同步replay，128实际长度全范围转换3823μs→89μs，8K 3827→329μs，32K 3813→1032μs，128K约3828→3829μs。host计时含replay/synchronize，非纯kernel与完整模型收益。当前正做eager/Graph全范围/Graph有效长度的完整模型匹配对照和GSM32回归；详见c8.md。
