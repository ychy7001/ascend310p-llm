# Qwen3.6-35B-A3B 310P 参数选择与华为评审记录

更新时间：2026-10-13  
硬件：Atlas 300I DUO 中单颗 Ascend 310P3，TP1，48 GB 级 NPU 内存  
模型：Qwen3.6-35B-A3B W4A8 实验 checkpoint；MTP 层保留浮点权重  

本文把当前已经在 a2 实机验证过的结果、仍待补齐的对照和产品参数建议分开记录。表中的吞吐是服务端阶段计时，通常为 warmup 后三轮中位数，不是正式 P95，也不应与另一台机器或不同 TP 配置的历史数字直接比较。

## 当前产品候选

112K 会议/对话档位建议先冻结为：

```text
TP=1
W4A8 routed experts
INT8 KV (C8), FP16 per-token/head scales
MTP=1 (num_speculative_tokens=1)
max_model_len=114688
gpu_memory_utilization=0.95
max_num_seqs=1
max_num_batched_tokens=1024
QWEN36_GROUPED_CHUNK_TOKENS=512
prefix cache: 先关闭，单独验收后再打开
Graph: 112K Graph mode=0 + 诊断性 updatable-graph 开关已跑通；尚不作为默认
```

这个档位满足超过 100K 的业务目标，并且已有真实启动、短请求、64K/128K 容量和长检索证据。`114688` 是配置上限，不等于每个请求都应该填满；业务默认可以使用 112K，保留输出预算后实际输入应相应减少。

近满上下文实测：eager 输入 `114560`、输出上限 128，实际输出 14，总耗时 432.24 s；Graph mode=0 同一 prompt IDs（SHA256 `5cd728...`）也正确检索到 `ASCEND732941`，实际输出 14，总耗时 431.94 s。各只有一个长样本，耗时包含长 Prefill 和生成，不能作为 P95 或稳定吞吐。Graph原始结果归档在 `data/ascend/experiment-results/112k-c8-force-graph/`，eager结果在 `data/ascend/experiment-results/112k-c8-eager/`。

## C8 与 C16/FP16 KV 的取舍

C8 的目的首先是容量和成本，而不是预设加速。当前 C8 后端把 K/V 写成 INT8、scale 写成 FP16，再在历史读取前转换到 FP16 NZ 后复用 310P 原生 PA；它不是已经证明的原生 INT8 PA 全链路。线性层仍使用 W4A8，MTP 权重仍是浮点。

在相同 W4、TP1、MTP1、Graph、32K 上限的匹配实验中：

| KV | 128 输入 Decode | 8192 输入 Decode | 32256 输入 Decode | Prefill | 观测内存峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C8 | 33.66 tok/s | 30.51 tok/s | 23.19 tok/s | 215/378/350 tok/s | 34238 MB |
| C16/FP16 | 33.75 tok/s | 32.29 tok/s | 26.70 tok/s | 215/379/352 tok/s | 36744 MB |

因此短请求几乎持平，8K 约下降 5.5%，近 32K 约下降 13.2%，Prefill 差异小于 1%；样本中约少用 2.5 GB。C8 的已知收益是长上下文余量，代价是历史越长，反量化/转换越影响 Decode。不能把“INT8 KV”写成固定加速结论。

补充了同一 `max_model_len=114688`、`.95`、W4/MTP1/eager、prefix off 的 FP16 KV 实测：8K/256 为 Prefill `381.61`、Decode `26.12 tok/s`；32256/256 为 Prefill `354.19`、Decode `22.30 tok/s`；近满 `114560` 输入检索耗时 `428.35s`。C8 同协议近满检索为 `432.24s`，两者均正确返回 `ASCEND732941`，差异约 0.9%。这说明当前 C8 在 112K 长检索上的主要价值仍是物理内存余量，而不是端到端速度提升。

当前 C16 的 256K 长请求尚未与 C8 112K 形成同协议完整对照。旧记录中的 310K/262K 是启动器的 KV 估算或不同配置容量，不应当直接当作已验证的长请求上限。后续应在同一服务协议下测 64K、112K、128K 和能启动的最大长度。

### a1 W8A8 参考点

a1 当前业务服务的真实启动参数为 TP2、W8A8、`max_model_len=262144`、`.90`、`max_num_batched_tokens=4096`、`FULL_DECODE_ONLY` Graph、prefix cache 开启、无 MTP。使用唯一 token 前缀隔离 cache 命中后，冷请求阶段计时为：8K Prefill 约 `1282 tok/s`、Decode `32.0 tok/s`；32K Prefill `1086.26 tok/s`、Decode `29.22 tok/s`。相同前缀再次请求会分别出现约 13.6K 和 64.7K tok/s 的缓存 Prefill，不能与冷请求混报。

这个 a1 结果适合做方向性 W8A8 参考，不能直接作为 a2 TP1/W4A8 的回归基线：硬件拓扑、TP、权重格式、Graph、batch、MTP 和 prefix-cache 状态均不同。后续优化顺序应先完成 a2 C8/Graph 生命周期和 MTP 门禁，再用固定 unique prompt、相同输出长度和阶段 counter 做跨机器矩阵对照；否则无法判断差距来自 W4 算子还是 TP/调度配置。

为避免 prefix cache 污染，`tools/bench_server_phases.py` 新增 `--unique-prompts`，让 warmup 和每一轮使用不同 token 前缀。A1 当前原服务（TP2/W8A8/Decode Graph/prefix cache enabled）使用该模式得到：8K/256 三轮 Prefill `1303.37/1304.36/1302.90` tok/s、Decode `32.16/32.16/32.14` tok/s；32K/256 前两轮有效 Prefill `1385.07/1383.88` tok/s、Decode `29.30/29.26` tok/s，第三轮因服务端阶段计数出现多请求增量而作废。8K 结果可作为新的冷请求方向性参考；32K 需要重新完成严格三轮后再用于正式比较。原始结果在 `data/ascend/experiment-results/a1-w8a8-reference/cold-unique-p8192-o256.jsonl` 和 `cold-unique-p32256-o256.jsonl`。

当前数据不能支持“TP2 Prefill 1000 tok/s，所以 TP1 必然 500 tok/s”的线性结论。a1 是 TP2/W8A8，a2 是 TP1/W4A8；TP1 还要承担 W4 权重解包/反量化、MoE routed-expert 调度和不同通信/同步开销。实测 a2 TP1 W4A8 冷 Prefill 在 8K/32K 约 `372--354 tok/s`（eager/MTP1）或 Graph 约 `380/352 tok/s`，确实低于 TP2 结果的一半，但差距不能归因于单一反量化步骤。需要用同一设备 profile 拆分 `QwenW4A8Int4MatmulV310`、专家 pack/dispatch、attention 和 host/Graph launch 后，才能判断下一步应优化算子还是调度。

### Prefill profile 的优化优先级

当前已有一份真实 310P3、TP1、W4A8 专家路径的 NPU profile（3072 输入、输出 1，单请求）。profile 的累计 kernel 时间会重叠，不能直接当作端到端墙钟占比；它用于排序热点，不用于宣称某个算子可以带来同等比例的加速。摘要归档在 `data/ascend/experiment-results/tp1-w4-profile/profile-summary.json`。

| 算子 | 次数 | 累计时间占比 | 备注 |
| --- | ---: | ---: | --- |
| `QwenW4A8Int4MatmulV310` | 360 | 21.51% | Cube median 约 92.29；是最大单一算子，但不像明显未启动 Cube |
| `MatMulV2` | 2168 | 19.17% | 与 W4 专家 MatMul 同量级，应检查非专家层和形状效率 |
| `hcom_allReduce_` | 648 | 18.23% | 通信/同步已经是主要项，TP 扩展和路由组织需要单独分析 |
| `TransData`、`Slice`、`IndexPut`、`Cast`、`ViewCopy` | - | 约 16% | 候选是布局、路由元数据和 activation pack 的融合或减少往返 |
| `QwenW4A8PackV310` | 360 | 1.51% | 只优化 pack 本身不可能解释 Prefill 与 A1 的全部差距 |

因此 Prefill 优化顺序暂定为：先确认相同输入下 profile 的墙钟阶段与 TP/通信开销，再检查 routed-expert 分组、激活 pack、布局转换和普通 MatMul；只有在这些成本隔离后，才继续改 W4 tile、L1 权重复用或跨 K 累加。A1 的 8K/32K 冷 Prefill（约 1282/1086 tok/s）保留为方向性 W8A8 参考；正式比较必须固定 unique prompt、输出长度、prefix-cache 状态、Graph capture 和 `max_num_batched_tokens`，并在 A2 上交替复测。

## Graph 修复探针（2026-10-12）

镜像中的 310P attention backend 声明自己属于 updatable graph，但 FULL graph 参数更新仍误进入通用 `update_graph_params`，造成 `graph_params=None`。新增 `QWEN36_C8_FORCE_UPDATABLE_GRAPH=1` 仅用于诊断：在 `direct_graph_310.py` 中将 Graph 参数更新判断切到 updatable 分支，默认关闭，不改变普通 eager/Graph 行为。

该开关已在真实 310P3 上通过三档上下文门禁：

| 配置 | 结果 |
| --- | --- |
| 4K、C8、MTP1、Graph、capture `[1,2]` | capture 成功；toy 8/8；A/B/A 一致；3072-token needle 正确；Prefill 384.20、Decode 32.69 tok/s |
| 32K、C8、MTP1、Graph、device-length、capture `[1,2]` | KV 124518 tokens；capture 成功；8K Prefill/Decode 379.72/30.67；近32K 351.53/23.47 tok/s；32256-token needle 正确（95.40s） |
| 112K、C8、MTP1、Graph、mode=0、device-length、capture `[1,2]` | 服务启动与capture成功；8K Prefill/Decode 379.05/21.08；32K 351.14/18.39 tok/s；114560-token needle正确（431.94s） |

112K Graph 需要 `COMPILATION_MODE=0`：mode=3 在 vLLM split-graph 过程中因节点仍有多个用户而报错。112K同一配置下Graph阶段数值与eager接近，但32K Decode约18.39 tok/s，不能据此宣称Graph加速；与早期eager数据协议并非完全交替匹配。该开关仍是实验性诊断路径，依赖monkeypatch和镜像版本，产品默认暂保留eager，待把修复收敛为可审查的backend改动并完成匹配A/B、MTP与回归门禁后再切换。

## MTP 0/1/2 的取舍

在 FP16 KV、eager、TP1、32K 上限、1024:512、同一 W4 checkpoint 的配对扫描中：

| 输入/输出 | MTP0 Decode | MTP1 Decode | MTP2 Decode | MTP1 Prefill | MTP2 Prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/256 | 未做同协议复测 | 26.77 | 29.97 | 194.84 | 187.91 |
| 8192/256 | 21.12 | 26.24 | 33.11 | 378.89 | 376.35 |
| 32256/256 | 18.83 | 22.46 | 28.85 | 352.08 | 350.24 |

MTP1 相对无 MTP 的 8K Decode 约提升 24%，近 32K 约提升 19%；Prefill 约持平或略降。MTP2 在这组重复/有利负载上比 MTP1 再快约 26%--28%，但会增加草稿计算、显存和 Graph 形状复杂度，接受率也依赖真实请求，不能用重复文本的 100% 接受率代表业务平均值。

C8+MTP2 eager 已经通过 112K 启动、8/8 toy、A/B/A 复用和 8K/32K 阶段计时：8K/128 Prefill `372.51`、Decode `31.57 tok/s`，32K/128 Prefill `346.04`、Decode `25.68 tok/s`。相对 C8+MTP1，Decode 分别约提升 20% 和 14%，但 Graph 仍在 310P attention 的 `update_graph_params` 阶段失败。因此产品候选仍选 MTP1；MTP2 保留为 eager 实验档，不能作为 Graph 默认。

补充的 C8+MTP2+Graph 4K 门禁（2026-10-13）已通过：使用`NUM_SPECULATIVE_TOKENS=2`、capture `[1,2,3]`、`COMPILATION_MODE=0`和诊断性updatable-graph开关，sanity 5/5、严格答案8/8、A/B/A一致、3072-token needle通过。3072/128 unique-prompt三轮Prefill中位`370.43 tok/s`、Decode中位`36.03 tok/s`。这是短上下文可运行证据；MTP2仍依赖实验开关，尚未完成8K/32K/112K容量、真实业务接受率和与MTP1交替复测，产品默认暂不切换。

## 长上下文与显存口径

vLLM 报告的 GPU KV cache size 是 scheduler 的聚合 block 容量，不是 C8 自己所有物理池的字节数，也不是当前请求已经使用的 token 数。C8 还要为 hybrid state、页对齐、scale 和共享 FP16 NZ scratch 保留空间。因此容量评估必须同时记录：启动配置上限、scheduler KV token 数、`C8_BUDGET`、实际请求长度、`npu-smi` 采样峰值和是否发生 OOM。

已有 C8 eager 证据：`.95` 下 64K 和 112K 可以启动；128K 需要 `.97`；`.99` 无法为 262K 提供足够物理空间。112K 的稳定性优先于把 memory utilization 推到 `.97/.99`。`0.90` 适合短/中上下文调试，`.95` 是当前产品候选，`.97` 仅作为专门的 128K 实验参数。

Prefix cache 的收益取决于请求是否复用完全相同的 token 前缀，不能把它计入冷请求容量或 Decode 吞吐。

补充的 C8/112K prefix-cache 实测使用 `max_num_batched_tokens=2048`（prefix cache 在该 hybrid 模式下要求 Mamba `block_size=1152 <= max_num_batched_tokens`；因此不能直接沿用默认 1024）：8K 冷请求 Prefill `361.45 tok/s`，后续相同 token IDs 的三次命中 Prefill `2263.83/2270.20/2274.52 tok/s`，Decode 约 `25.1 tok/s`。prefix cache counter 从 0 增至每次 6912 命中 tokens，服务日志命中率最高约 63.3%。这证明重复前缀能显著降低 Prefill，但没有带来 Decode 加速；建议作为会议/模板前缀明确重复时的可选配置，并把 `2048` 约束写入启动检查。

112K 长前缀复用也已验证：114560-token 冷请求耗时 `412.00s`、Prefill `278.78 tok/s`；同一 token 前缀再次请求耗时 `9.33s`、Prefill `13846.95 tok/s`，Decode 约 `12.51 tok/s`，两次均生成 14 tokens。长请求 prefix hit counter 增至 `140544`。长前缀命中可以显著降低重复会议材料的 TTFT，但不改善历史 KV 主导的 Decode，因此产品上建议把 prefix cache 作为有明确模板/会议前缀复用时的可选档位。

## 当前代码入口与简化边界

实验后端集中在 `experimental/c8_vllm_adapter.py`，代码入口只有四类：

1. `get_spec`：把目标 full-attention 层替换为 `C8FullAttentionSpec`，并拒绝 TP2、KV transfer 和未授权 MTP2。
2. `get_configs`：计算 scheduler 的保守 block 数，并打印 `C8_BUDGET`。
3. `allocate`：在 310P runner 上分配 K/V INT8、scale FP16 四张量，保留原 scheduler charge。
4. `forward`：写入 C8 cache，按实际长度或 Graph 固定长度转换到 FP16 NZ，再调用原生 PA/split-fuse。

这些 monkey patch 只在 `QWEN36_C8_ENABLED=1` 的实验启动器中加载，默认 W8A8/W4A8 vLLM-Ascend 行为不变。给华为评审时应优先把环境门禁、容量计算和 attention forward 拆成小函数，并为每个函数保留输入/输出契约；不要把 C8 patch 混入官方默认路径，也不要先做大规模重构。当前最值得先改的是重复的环境判断、无名常量（64、16、256）和错误信息，让硬件限制直接出现在代码中。

## 给华为工程师的待确认问题

- 310P3 是否有可公开使用的 INT8 KV PA/Block Table 路径，能否避免当前 C8 -> FP16 NZ 转换？
- `graph_params` 在 310P attention 的 FULL graph/MTP capture 中应由哪个 runner 生命周期初始化？
- 310P 上 W4A8 的 `mad_s4`/GMM 推荐 tile、group size 和 routed-expert 小 batch 调度参数是什么？
- C8 的 scale 布局、页对齐和 hybrid state 是否有官方 allocator 约束，可否减少当前 padding？
- MTP2 在 310P Graph 下推荐的 capture sizes、rollback 和 KV cache 形状是什么？

源码调查补充：当前上游 310P `AscendAttentionBackendImpl310.update_graph_params` 直接声明 `NotImplementedError("FIA and PA should be use UpdatableGraph.")`，而 FULL graph 更新流程仍会通过 `impl_cls.update_graph_params` 触达该接口。实验日志中的 `graph_params=None` 因此应先按 310P attention 的 graph 类型/生命周期不匹配排查，不能用跳过更新或强行填全局对象的方式掩盖问题；这也是提交上游前需要请华为确认的具体代码点。

原始日志和结果保留在 a2 的 `qwen36-public-release-20261012` 目录；本仓库只提交复现脚本、汇总和不含权重的代码。所有尚未完成的项目都标为实验性，不把启动成功等同于性能或精度验收。
