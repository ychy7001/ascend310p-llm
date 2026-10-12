# Ascend 310P3 · Qwen3.6-35B-A3B
# W4A8 / MTP / KV-C8 适配与初版测试报告

技术评审版 v0.2 · 2026-10-12 · 联系人：ychy7001

## 01　结论与评审目标

本项目在 Atlas 300I DUO 的真实 Ascend 310P3 上，已完成 Qwen3.6-35B-A3B Routed Experts W4A8 的 TP1/TP2 启动和生成，以及使用模型自身 BF16 MTP 的推测解码适配。标准启动器使用 FP16 KV；实验 C8 已完成 TP1/eager/MTP=0/4K 完整模型短生成与 chunked Prefill。W4A8、MTP、C8 三项联合推理尚未验收。

目标是同时研究模型权重与 KV 内存压缩、单流 Prefill/Decode、并发吞吐和较长上下文，最终维持合理精度并验证是否有稳定收益。历史约 25 tok/s Decode、1000 tok/s Prefill 仅为观察值，不能直接用作严格对照。

| 项目 | 初版证据 | 当前边界 |
| --- | --- | --- |
| W4A8 | Routed Experts 完整模型 TP1/TP2 生成 | 其余层保留浮点；尚无全面质量结论 |
| MTP | Qwen3.6 自身 19 个 BF16 张量；TP1/TP2 生成 | 已测 speculative tokens=1 |
| 长上下文 | 无 MTP TP1 实际 32K、128K prompt 生成 | 256K 配置启动不等于实际 256K 验收 |
| C8 | 独立算子与 TP1 完整模型短生成；匹配 FP16 KV 对照 | 速度下降；正式质量、容量、长上下文待验收 |
| 公开交付 | GitHub 源码、自包含 checkpoint；ModelScope 64 文件远端哈希通过 | 未做全量重新下载后的推理验收 |

工程判断：W4 路径已值得继续研究，但未证明稳定 10%–20% 的端到端改善。C8 的主要价值仍是容量；当前转换成本高于原生 FP16 路径，需要接口、布局和流水方面的指导。

本报告请求华为确认具体 310P3 算子能力、C8/NZ 布局、混合计算与 hybrid cache 分配方法。该实现是社区基础上的实验适配，不是华为、Qwen 或 vLLM 官方认证版本。

证据标记：E＝真实设备实验；S＝源码检查；H＝尚待验证的推断。结果表为 E，适配逻辑为 S；不将 API 名称或单次编译失败扩大为完整硬件能力结论。

## 02　环境、模型与证据范围

实验机器使用一张 Atlas 300I DUO、两颗 Ascend 310P3。npu-smi 可用设备内存约 44,277 / 43,693 MB，低于标称每颗 48 GB；容量规划采用设备实际可用值。a2 专用于实验，服务顺序运行，避免两容器争用同一 NPU。a1 为已有 W8A8 TP2 服务参考，不改动其部署。

| 组件 | 实际使用版本 / 来源 |
| --- | --- |
| Driver / npu-smi | 26.1.1；Firmware 独立版本未在本交付中锁定 |
| 推理镜像 | quay.io/ascend/vllm-ascend:v0.23.0.post1-310p |
| CANN / PyTorch | CANN 9.1.0；Torch 2.10.0+cpu |
| torch-npu | 2.10.0.post4 |
| vLLM 基础提交 | 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 |
| vLLM-Ascend 基础提交 | 1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8 |
| 社区 W4 基础提交 | opensensor/vllm-ascend：0637d3baa6bc2000ba6433b301f233c6516847b4 |
| 本报告源码快照 | ychy7001/ascend310p-qwen36：3688ebb |

源模型为 ModelScope Qwen/Qwen3.6-35B-A3B BF16 checkpoint。社区 Qwen3.8-Flash-Next 的 310P 实现提供算子思路和部分代码基础；模型、专家 shape、路由、加载与 MTP 布局存在差异，未假定其 checkpoint 或整套框架可直接复用。

原始源权重的 immutable revision 记录不完整，这是当前追溯缺口。公开制品以逐文件 SHA256 固定；后续重新量化应先锁源 revision、文件哈希和 ModelSlim 版本。不能以当前远端 main 代替已用源 revision。

测试分三层：①独立算子数值、warmup/repeats 和 device profile；②完整模型启动与短生成；③初版 HTTP streaming 对照。三层证据不可替代：基本生成不证明准确率，kernel 累计时间不证明墙钟加速，容量估算不证明长上下文质量。

## 03　W4A8 量化与自定义算子路径

适配流程：原始 BF16 Routed Experts → ModelSlim SSZ 对称 INT4 → 输入轴 nibble 打包与 scale/offset → 自定义权重加载与专家路由 → 设备端 per-token A8 → INT4 limbs Cube 计算 → 浮点输出。仅替换 Routed Experts；共享专家、Attention、Embedding、LM head 和 vision 等保留原始浮点张量。

| 模型项目 | 真实形状 / 契约 |
| --- | --- |
| 主模型 Routed Experts | 40 层 × 256 experts × 3 projections |
| Gate / Up 合并 | 输入 K=2048，输出 N=1024；权重逻辑 [1024,2048] |
| Down | 输入 K=512，输出 N=2048；权重逻辑 [2048,512] |
| 量化粒度 | 对称 per-output-channel SSZ；signed INT4 |
| 权重 scale / offset | FP16 scale；零 offset；scale 重复到 G128 ABI |
| 激活 | 设备端 per-token INT8；两组 signed INT4 limbs |

G128 是当前 kernel 的布局契约，首版权重不是每个 128 元素组独立拟合 scale。转换适配器调用 ModelSlim WeightPerChannelSsz，避免 W8→W4 重量化；保留源权重，输出独立 sidecar，再导出自包含 checkpoint。

S：native_int4_schedule.h 使用 int4b_t 的 Mmad，底层路径对应 mad_s4。INT8 激活通过低/高 INT4 limbs 与修正项组合，包含高 limb 的 ×16 系数；per-channel/per-token scale 在整数累加后应用。E：这些算子已在 310P3 编译并被真实模型使用。证据支持可用 INT4×INT4 组合路径，不等于官方提供直接 INT8×INT4 单指令。

三个独立 OPP 家族：qwen_w4_a8_int4_matmul_v310、qwen_w4_a8_pack_v310、qwen_w4_a8_swiglu_pack_v310。Torch extension 使用独立 qwen36_int4 namespace，不覆盖官方 _C_ascend。Qwen3.6 adapter 处理专家替换、权重加载与 TP 所有权；首版未默认启用实验性 fused down-reduce。

W4A16、W4 存储后解包 INT8 计算仍是对照路线，尚无完整真实模型同条件结果。W4A16 可能增加反量化计算；W4→INT8 的全局展开会侵蚀存储和带宽收益，必须将解包计入端到端时延。

## 04　MTP、TP1/TP2 与上下文容量

原始 Qwen3.6 BF16 checkpoint 已有 19 个 mtp.* 张量，包括 stacked expert gate_up_proj [256,1024,2048] 和 down_proj [256,2048,512]。无需借用 Qwen3.5 MTP。适配期间曾漏列两个 stacked 张量，已补全；公开 checkpoint 的结构检查确认全部 19 个 BF16 张量存在。

MTP drafter 的 decoder 类型检查与目标 W4 adapter 不兼容；补丁让 MTP 前缀保留浮点 decoder，避免目标 W4 初始化误作用于 drafter。日志确认 target/draft 共享 embedding。仅 num_speculative_tokens=1 已完成设备验证。

| 配置 | 初版加载 / KV 观察 | 验证范围 |
| --- | --- | --- |
| TP1，无 MTP，262144/.90 | KV 4.95 GiB，估计最大 255744 | 启动因 KV 不足失败 |
| TP1，无 MTP，262144/.95 | 权重约 23.82 GiB；KV 6.03 GiB | 启动成功；实际 32K/128K prompt 生成 |
| TP1，MTP=1，131072/.95 | KV 约 4.30 GiB；估计 193331 tokens | 启动与短生成；非实际 128K MTP 验收 |
| TP2，MTP=1，262144/.90 | 每卡权重约 13.73 GiB；KV 10.41–10.48 GiB | 启动与短流 / 并发 4 |

TP1 精确 32768 prompt、输出 8 tokens：92.18 s；131072 prompt、输出 8 tokens：484.77 s，usage 与输入长度一致。两项证明请求可完成，不证明长上下文任务质量；耗时包含 HTTP、Prefill 和生成，不能直接视为纯 Prefill tok/s。

TP1 单流 p128/o256，无 MTP：20.9767 / 20.9682 / 20.9857 tok/s；TTFT 约 0.595 s，平均 TPOT 约 45.54 ms。约 .90 的参数不是所有配置的最优值；.95 是本轮 TP1 容量门禁选择，需重新测峰值与稳定性。

MTP 增加权重和工作区、缩小 KV 余量。短流接受率会随内容变化；单次接近 100% 不代表长期收益。后续必须检验 draft/verify 有效长度、拒绝 token 后 slot 覆盖及 linear state 回退，再与 C8 组合。

## 05　初版性能与公开产物验证

TP2、FP16 KV、eager、max_model_len=262144、memory=.90；temperature=0、固定 token IDs、ignore_eos。每组 1 次 warmup 后 3 轮，指标为输出 tokens / 请求墙钟时间，含 Prefill、HTTP 和调度；不能直接命名为纯 Decode tok/s。

| 负载 | 无 MTP | MTP=1 | 初版观察 |
| --- | --- | --- | --- |
| p128/o256，c1 | 约 26.05 tok/s | 27.69–28.49 tok/s | 平均约 +7.9%，探索结果 |
| p128/o256，c4 | 约 84.66 tok/s | 87.36–96.51 tok/s | 平均约 +7.9%，有轮间波动 |
| p512/o256，c1 | 中位数 24.8856 | 中位数 25.7386 | 中位数 +3.43% |
| p512/o256，c4 | 中位数 71.0115 | 中位数 73.6933 | 中位数 +3.78% |

p512 的两组 prompt IDs 已核对一致。c1 TTFT 中位数：0.8765→0.9440 s；请求平均 TPOT 中位数：36.90→35.30 ms。收益仍未证明稳定 10%–20%；MTP 自动改变的调度默认值属于系统行为，未人为强行对齐。

单请求 profile，rank0 专家 kernel 累计时间：

| 阶段 | W8 Gate/Up + Down | W4 Gate/Up + Down |
| --- | --- | --- |
| Prefill | 43.104 ms | 108.750 ms |
| Decode | 32.504 ms | 22.766 ms |

W4 Decode 专家更快，Prefill 更慢；AllReduce 等待和 TransData 也增加。两侧专家组织、量化范围不同；profile 的 W4 为 fused 候选，不等同公开普通路径。累计时间可重叠，不能转换成严格墙钟占比。该固定 token 工作负载的 W8 文本为空但 usage 有效，不是质量证据；尚无实测整机 GB/s，不能下定论为 memory-bound。

公开 checkpoint：52 分片、93125 indexed tensors、92160 expert weight/scale/offset tensors；tensor payload 24591896288 bytes，文件约 24.63 GB。无软链接，结构与逐文件 SHA256 校验通过。新构建 OPP/extension 的 TP1/MTP=1/4K/.90 服务通过 5 类生成 sanity，p128/o64 三轮约 21.94–21.97 tok/s。数学被 128 输出上限截断，sanity 不代替准确率。

## 06　KV-C8 的路径与容量设计

完整模型 C8 生命周期：FP16 K/V 输入 → 设备端动态 per-token/head scale → INT8 ND 持久缓存与 FP16 padded scale → 按设备页表选择有效页 → 临时 FP16 NZ → 原生 310P Paged Attention。linear attention recurrent state 保持浮点。该方案不依赖静态 KV 标定权重。

| 项目 | 实际契约 |
| --- | --- |
| 目标 Attention | 16 query heads / 2 KV heads / dim256 |
| 持久 K/V | INT8 [slots, KV heads, dim] |
| scale | FP16 [slots,16]；32-byte 行对齐 |
| 原生 PA 临时 NZ | [blocks, KV heads×dim/16, block,16] |
| 已测基本形状 | block64/dim256；block128/dim128 |

量化使用实际落盘的 FP16 scale：amax/127，最小值钳制到 FP16 normal tiny，再 round/clamp 至 [-127,127]。scatter 支持有效 slot、padding slot=-1；decode append、乱序页、重复页和动态 Graph 已做数值验证。invalid 页 zero-fill 仅为内存保护，不是有效历史语义。

32K、2 KV、dim256 的持久 C8 为 34 MiB，FP16 K/V 为 64 MiB，张量级减少约 46.9%。staged 路线另需 64 MiB 临时 NZ。单层同时存在 C8 与 NZ 时峰值更大；只有跨层复用工作区、移除参考副本并计入并发/Graph/allocator，才可能有全模型净收益。

S：hybrid spec 将 Attention page 对齐到 Mamba state page；310P 的 NPUModelRunner310._allocate_kv_cache_tensors 覆写通用分配与 reshape，必须接入专用路径。实验后端使用真实 INT8 spec、显式 scale 计费、独立层分配和跨层共用 NZ；保留上游 page 预算并减少 block 数，不绕过分组检查。

E：TP1/MTP=0/eager/4096/.90、max-seqs=1、max-batch=256、无 prefix cache，五类生成 sanity 通过；completion tokens=64/64/35/64/64。无历史 Prefill 消费当前浮点 K/V；历史读取使用真实 INT8 cache。linear state 保留浮点。十个 full attention 层确认 INT8 K/V、FP16 scale。

初版原预算 5,308,416,000 bytes，40 层独立分配 5,284,823,040 bytes，scheduler blocks=56，共用 NZ=8,388,608 bytes。每 full 层 64512 slots，有效 payload 70,189,056 bytes、allocation 132,120,576 bytes；每层仍有 61,931,520 bytes 无用尾部。紧凑分配已在真实设备通过五类生成与两组 benchmark，十层物理 payload 共减少 619,315,200 bytes（590.625 MiB），scheduler 仍为 56 blocks，不宣传新增上下文容量。紧凑版 p512/o64、p3072/o32 中位吞吐为 13.4352 / 2.8361 tok/s，仍慢于 FP16 KV；整模型峰值和长上下文收益待验收。

另一条候选为融合 Cube QK/PV、Split-KV online softmax，已通过独立数值与 Graph，但长序列仍较慢。基础 Mmad<float,half,int8_t> 在当前 SDK 编译失败，而 half/half 控制组成功；高阶路径见 UB antiquant。该证据仅说明当前 API/实现限制，未证明底层全部硬件能力。

## 07　C8 初版数据与失败证据

真实 310P3，固定 shape/seed，5 次 warmup、20 次测量；host latency 含 launch+synchronize，device latency 为 profiler 采集，两者分开报告。原始 samples 保留于实验机。

| 优化对照 | 4K 完整转换+PA | 32K 完整转换+PA |
| --- | --- | --- |
| sync tile32，第 1 轮 | 312.939 μs | 2166.137 μs |
| pipeline tile32，第 1 轮 | 310.203 μs | 1975.600 μs |
| sync tile32，第 2 轮 | 302.966 μs | 2036.914 μs |
| pipeline tile32，第 2 轮 | 295.772 μs | 1843.385 μs |

同 tile32 长序列下降 8.8%/9.5%；4K 仅 0.9%/2.4%，不认定稳定收益。双缓冲增加 UB 占用，width512 实际 tile 从 64 降到 32，不能只比较配置名。32K device 纯转换均值：sync32 1122.061 μs、pipeline32 979.078 μs；scalar ratio .416→.755，事件控制成本增加。ratio 可重叠，不能相加或推导带宽占比。

此前 compact-scale tile64 纯转换约 923.863 μs，仍优于本轮 pipeline32；这是跨阶段观察。同期 pipeline 完整路径 4K/32K 约 292.860/1969.171 μs，原生 FP16 PA 约 232.306/877.252 μs；当前 C8 仍慢约 1.26/2.24 倍，未进入默认服务。

| 失败 / 限制 | 具体证据与处理 |
| --- | --- |
| 官方 int8_per_token_head 启动 | hybrid page unification：page size not divisible by maximum page size |
| 跳过 linear / 关闭 hybrid manager | 仍失败；关闭 manager 后不能统一 cache spec |
| 静态 C8 权重契约 | 当前 checkpoint 的 kv_cache_scale / kv_scale tensor 数为 0 |
| Cube float→half CAST_RINT | 310P 失败；改 CAST_NONE 后数值门禁通过 |
| 混合基础 Mmad 类型 | half×int8 编译失败；同 shape half×half 编译成功 |

匹配完整模型对照：同 checkpoint/OPP/extension，TP1/eager/MTP=0/4K/.90、max-seqs1/max-batch256，prompt IDs 相同。1 warmup + 3 rounds，中位数如下；吞吐含 Prefill/HTTP，不是纯 Decode 指标。

| 输入 / 输出 | FP16 KV / C8 tok/s | FP16 / C8 TTFT s | FP16 / C8 TPOT ms |
| --- | --- | --- | --- |
| p512/o64 | 14.2881 / 13.2868 | 1.6193 / 1.6781 | 45.40 / 49.80 |
| p3072/o32 | 2.9583 / 2.8152 | 9.3947 / 9.7878 | 45.90 / 50.91 |

当前 C8 吞吐下降 7.01% / 4.84%，平均 TPOT 增加 9.69% / 10.91%。两边五类 sanity 均通过，code/math/repeat 文本相同，英文/中文不同；不证明质量等价。连续请求覆盖页释放再分配，尚无严格请求隔离、logits 或正式质量差分。

独立 mutable query/table/scale 的 Graph 回放已通过；C8 多 query 因果参考 Q=2/3 最大绝对误差约 0.00547–0.02219。它们未证明 vLLM scheduler、MTP 拒绝回退或模型精度。当前无法通过仅增加启动参数获得已验收 C8。

## 08　希望华为协助的具体事项

1. 310P3 / CANN 9.1 的 C8 Paged Attention 与 cache 写入：是否有可用接口或参考实现？请明确产品支持矩阵、ND/NZ、scale 粒度、stride、page size、最小 shape；重点为 16Q/2KV/dim256/block64。

2. 混合矩阵能力：half×int8 是否必须经过 UB antiquant？是否存在避免 FP16 全局临时 tile 的路径？希望区分基础 Mmad、高阶 Matmul、SDK 限制和硬件潜在能力，不以通用 INT8/INT4 文档名称替代 shape/布局验证。

3. C8→NZ 的布局与流水：如何降低两次转置、事件与 scalar 开销，同时维持较大 tile？310P3 的 UB→L1、NZ 搬运和 MTE 并行有哪些可用或软件模拟边界？希望提供本 shape 的 tiling / 示例。

4. W4 Routed Experts：Gate/Up K2048/N1024、Down K512/N2048；Decode 小 M 与 Prefill routed 分组应如何优化？希望指导 mad_s4 跨 K 累加、L1 权重复用、SwiGLU/activation pack 融合；明确不能从 910B/A2/A3 直接迁移的特性。

5. Qwen3.6 hybrid cache：full/linear attention 的动态 scale、Mamba page padding、分组和共享 scratch 如何联合设计？MTP draft/verify 拒绝后，哪些 cache/state 需要回退，哪些仅更新有效长度即可？希望获得正确性约束与参考实现。

下一验收节点：在已通过的 TP1 短生成、chunked Prefill 基础上验证紧凑物理分配、请求隔离、logits/生成质量与峰值内存；查明转换与调度成本。随后再加 MTP=1、TP2、并发 2/4/8、Graph、32K/128K/256K 和全面质量。

研究退出边界：若路线只有容量收益而速度下降，报告真实权衡；若 SDK/hybrid 分配受阻，提交精确最小复现，不通过隐藏 FP16 副本或绕过检查伪造 C8 成功。正式门禁需同机同模型同输入/输出/并发的 W8A8/W4 对照、TTFT/TPOT P50/P95、峰值内存及质量集合。

## 09　代码、复现与证据交付

公开代码：https://github.com/ychy7001/ascend310p-qwen36

权重：https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P

首版报告对应 3688ebb；后续实验补充见下节和docs/c8.md。原始作者许可、版权及来源提交见 NOTICE / docs/provenance.md，初始runtime/native快照见 docs/runtime-source-sha256.json，后续实验adapter修改由Git提交追踪。社区模型卡仅作思路参考，不声明直接复现成功。

| 复现步骤 | 仓库入口 |
| --- | --- |
| 编译三类 OPP 与独立 extension | scripts/build-native.sh；README 固定镜像、CANN 环境与 Triton 准备 |
| 源浮点 → SSZ sidecar | tools/convert_modelslim_w4a8.py / modelslim_ssz_adapter.py |
| 自包含 checkpoint 导出 / 校验 | tools/export_checkpoint.py / validate_checkpoint.py |
| TP1/TP2、MTP 开关启动 | scripts/run-service.sh；默认 TP1/4K/.90/eager |
| 生成 sanity / streaming | tools/bench_generation_sanity.py / bench_http_stream.py |
| C8 编译、数值、benchmark | docs/c8.md；experimental/ascend_c8_kernel |
| 发布与新构建验证 | docs/release-validation.md |

典型启动：设置 MODEL_DIR、OPP_DIR、EXTENSION 的绝对路径；TP=1 MTP=1 MAX_MODEL_LEN=4096 MEMORY_UTILIZATION=0.90，执行 scripts/run-service.sh。TP2 改 TP=2。尚无已验收 C8 服务开关。

公开源码对应的三类 OPP 和 extension 已在新目录分步编译、打包、安装并完成生成检查；过程中补回准备脚本误过滤的源码补丁。没有逐字端到端执行公开 build-native.sh，不声称新下载环境完整脚本验收。ModelScope 约 24.63 GB、64 文件已于北京时间 12:10:37 完成提交，失败 0、耗时约 61 分钟；64 文件远端大小与 SHA256 全部匹配。新下载 config/index/deployment-sha256 三文件逐字节匹配；没有全量重新下载权重做推理。模型卡随后独立更新 C8 状态，不改动权重。

原始构建、服务、请求、profiler 和数值 samples 留在实验机，不提交 Git。公开交付提供可复现源码与汇总；华为需要时可另按指定范围提供原始日志。部分旧实验目录后缀为 20261013，实际时间以日志为准，本报告统一使用 2026-10-12。

待补完整门禁：group size 对照、W4A16/解包 INT8 对照、正式质量集合、c2/c8 与尾延迟矩阵、设备级 Prefill/Decode 指标、实际近 256K 请求、C8 长上下文/质量/容量及联合稳定性。用户已转发冻结版v0.2 DOCX/PDF，后续Markdown更新不会自动同步其中数据。

## 10　后续联合 Graph / 长上下文实验补充（2026-10-12）

E：W4A8+自身BF16 MTP=1+动态C8+Decode Graph，在TP1/4K/.90、max-seqs1/max-batch256、关闭prefix cache配置完成capture/replay。模式为编译mode0与FULL_DECODE_ONLY/[1,2]；实验补丁统一GraphParams初始化条件，未跳过attention参数更新。图内C8固定转换max-context范围，原生PA按设备有效长度遮罩。之前使用CPU长度裁剪的候选出现3072-token检索失败，已淘汰；修正候选相同检索通过，固定GSM32为32/32，0解析缺失/0截断，8toy和A/B/A复用检查通过。小样本不等于全面质量验收。

服务端请求阶段计时（1warmup/3次、独占端点、逐次counter增量核对1）：

| 配置与负载 | Prefill中位tok/s | Decode中位tok/s |
| --- | --- | --- |
| 4K上限、p128/o256 | 未用于长Prefill指标 | 32.9694 |
| 4K上限、p3072/o1 | 318.401 | 无独立后续Decode |
| 4K上限、p4032/o32 | 317.689 | 31.1525 |
| 128K上限/.95、p128/o64 | 211.883 | 20.6136 |

指标是vLLM服务端请求阶段墙钟时间，非纯kernel时间；Decode按`(completion_tokens-1)/decode_seconds`。W8历史约1000tok/s来自TP2及不同Prefill分块，尚无严格同条件对照。4K配置速度不能套用到128K配置。

E：128K上限/.95联合Graph服务初始化成功：C8 115调度页、共享NZ 256MiB、Graph约0.35GiB；精确65,536输入tokens中间needle检索通过，14输出tokens，总耗时269.763s。单次服务端Prefill268.772s（243.835tok/s）、Decode0.918615s（14.1517tok/s），没有重复统计。130,560输入请求则持续等待capacity，未执行也未产生生成结果。旧门禁只计Attention页、遗漏其他hybrid组和调度预留；新保守门禁与`.97`候选正在验证。**目前已验收的联合长请求为64K单次，不是128K/256K产品指标。**

S：当前固定max-context转换条件也覆盖eager chunked Prefill，可能增加长配置成本。待验证补丁将固定范围限定于FULL图forward，Prefill恢复按实际历史长度转换；之后扫描batch256/512/1024/4096并补齐同机同TP W8对照。Prefill专家大M路径仍是主要优化方向，不能假定W4权重压缩自然带来Prefill加速。

完整协议、原始文件名和复现入口见docs/c8.md、tools/bench_server_phases.py。冻结版Office报告尚未包含本节。
