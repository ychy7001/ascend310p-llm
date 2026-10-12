# Ascend 310P Qwen3.6 低比特推理：技术反馈

2026-10-12，北京时间。硬件实验使用Atlas 300I DUO上的310P3；以下为研究结果，不是华为或Qwen官方认证实现。

## 可复现状态

| 技术 | 当前状态 | 验收边界 |
| --- | --- | --- |
| Routed Experts W4A8 | 完整模型TP1/TP2启动、生成 | 其它层保持浮点；不保证优于W8A8 |
| W4A8+MTP | Qwen3.6自身BF16 MTP，TP1/TP2生成与部分benchmark | 仅num_speculative_tokens=1已测 |
| C8 | 独立page写入、读取、反量化、Attention与mutable Graph已测 | hybrid model backend尚未接入 |
| W4A8+MTP+C8 | 尚未联合运行 | 不属于首版已支持配置 |

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

C8先实现TP1/eager/单请求短生成，关闭MTP，仅压缩full-attention K/V，linear-attention状态维持原精度。按层复用NZ scratch；验证chunked prefill、decode append、页复用、生成质量与实际峰值内存，再加MTP拒绝token/state rollback，最后TP2/并发/Graph/长上下文。
