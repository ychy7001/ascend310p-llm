# Qwen3.6-35B-A3B 310P 参数选择与华为评审记录

更新时间：2026-10-12  
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
Graph: 112K 当前使用 eager；>32K Graph 生命周期问题修复前不作为默认
```

这个档位满足超过 100K 的业务目标，并且已有真实启动、短请求、64K/128K 容量和长检索证据。`114688` 是配置上限，不等于每个请求都应该填满；业务默认可以使用 112K，保留输出预算后实际输入应相应减少。

近满上下文实测（2026-10-12）：输入 `114560`、输出上限 128，实际输出 14；总耗时 432.24 s，检索词正确返回，服务端 MTP 平均接受长度约 1.85。该结果证明 112K 档位可以完成一次真实长检索且没有 OOM；耗时包含长 Prefill 和生成，只有一个样本，不能作为 P95 或稳定 Decode 吞吐。原始结果归档在 `data/ascend/experiment-results/112k-c8-eager/`，完整 a2 日志留在实验机 release 目录。

## C8 与 C16/FP16 KV 的取舍

C8 的目的首先是容量和成本，而不是预设加速。当前 C8 后端把 K/V 写成 INT8、scale 写成 FP16，再在历史读取前转换到 FP16 NZ 后复用 310P 原生 PA；它不是已经证明的原生 INT8 PA 全链路。线性层仍使用 W4A8，MTP 权重仍是浮点。

在相同 W4、TP1、MTP1、Graph、32K 上限的匹配实验中：

| KV | 128 输入 Decode | 8192 输入 Decode | 32256 输入 Decode | Prefill | 观测内存峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C8 | 33.66 tok/s | 30.51 tok/s | 23.19 tok/s | 215/378/350 tok/s | 34238 MB |
| C16/FP16 | 33.75 tok/s | 32.29 tok/s | 26.70 tok/s | 215/379/352 tok/s | 36744 MB |

因此短请求几乎持平，8K 约下降 5.5%，近 32K 约下降 13.2%，Prefill 差异小于 1%；样本中约少用 2.5 GB。C8 的已知收益是长上下文余量，代价是历史越长，反量化/转换越影响 Decode。不能把“INT8 KV”写成固定加速结论。

当前 C16 的 256K 长请求尚未与 C8 112K 形成同协议完整对照。旧记录中的 310K/262K 是启动器的 KV 估算或不同配置容量，不应当直接当作已验证的长请求上限。后续应在同一服务协议下测 64K、112K、128K 和能启动的最大长度。

## MTP 0/1/2 的取舍

在 FP16 KV、eager、TP1、32K 上限、1024:512、同一 W4 checkpoint 的配对扫描中：

| 输入/输出 | MTP0 Decode | MTP1 Decode | MTP2 Decode | MTP1 Prefill | MTP2 Prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/256 | 未做同协议复测 | 26.77 | 29.97 | 194.84 | 187.91 |
| 8192/256 | 21.12 | 26.24 | 33.11 | 378.89 | 376.35 |
| 32256/256 | 18.83 | 22.46 | 28.85 | 352.08 | 350.24 |

MTP1 相对无 MTP 的 8K Decode 约提升 24%，近 32K 约提升 19%；Prefill 约持平或略降。MTP2 在这组重复/有利负载上比 MTP1 再快约 26%--28%，但会增加草稿计算、显存和 Graph 形状复杂度，接受率也依赖真实请求，不能用重复文本的 100% 接受率代表业务平均值。

C8+MTP2 eager 已经通过 4K 启动和 8/8 toy 正确性；C8+MTP2 Graph 在 310P attention 的 `update_graph_params` 阶段仍因 `graph_params=None` 失败。故当前产品候选选 MTP1；MTP2 保留为实验开关，修复 Graph 参数生命周期、完成 112K 质量和长 Decode 后再决定。

## 长上下文与显存口径

vLLM 报告的 GPU KV cache size 是 scheduler 的聚合 block 容量，不是 C8 自己所有物理池的字节数，也不是当前请求已经使用的 token 数。C8 还要为 hybrid state、页对齐、scale 和共享 FP16 NZ scratch 保留空间。因此容量评估必须同时记录：启动配置上限、scheduler KV token 数、`C8_BUDGET`、实际请求长度、`npu-smi` 采样峰值和是否发生 OOM。

已有 C8 eager 证据：`.95` 下 64K 和 112K 可以启动；128K 需要 `.97`；`.99` 无法为 262K 提供足够物理空间。112K 的稳定性优先于把 memory utilization 推到 `.97/.99`。`0.90` 适合短/中上下文调试，`.95` 是当前产品候选，`.97` 仅作为专门的 128K 实验参数。

Prefix cache 可能降低重复会议前缀的实际压力，但当前主要数据是在 prefix cache off 下获得的，不能提前把它计入容量或吞吐收益。下一轮单独做 prefix hit/miss、首请求/复用请求和长输入 Decode 对照。

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
