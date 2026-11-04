# GDN 上游候选评估与当前取舍

日期：2026-11-04。用户补充研究作为线索，采用“公开代码审计、隔离验证、选择性采用、增量贡献”的路线。当前继续固定版本验证，不将所有候选依次移植。官方GitHub API元数据、patch与review快照保存于 `results/upstream-gdn-pr-audit-20261104/`；状态只代表查询时刻。

## 当前决策

| 候选 | 决策 | 依据与最小后续范围 |
|---|---|---|
| #17640，head `18163f6b1d208ab3b4fb06e9752225611c37f40e` | 保留隔离诊断；暂不采用 | 已直接复用原生WY；完整配套H/O算子门禁通过，但35B业务语义门禁失败。先贡献生产形状回归及最小复现，停止扩大独立WY实现 |
| #18057，head `2620e77d7bdb3c54c6ea98b24b570269f6eed6b7` | 下一候选，暂不开始迁移 | 接入外部CANN WY/gating，未替换完整H/O。生产9.1-beta，PR Docker为9.2.0；先核对匹配的run/wheel、依赖和API，再决定单独WY验证，gating另作变量 |
| #18410，head `988fa61e81b9…` | 备选，当前不叠加 | 初始化时绑定算子，与#18057 dispatch路线择一，避免两套加载/回退同时进入生产 |
| #11941 / #13122 / #17412 | 参考，不重复实现 | 原生WY及CANN集成已有重叠；保存代码差异，只有独立缺陷/支持范围/收益证据才考虑独立PR |
| #17604，head `630e2729d7e2e0476373e8c72010015fec940a33` | P1参考；310P兼容未证明 | 融合QKV整理、L2Norm与gating的Triton prepare。改动通用ops/triton和运行时patch，未提供本轮310P生产栈执行证据；参考310P gating函数不等于支持310P部署。暂不直接移植或引用作者Decode收益为本机收益 |
| #8872 MegaGDN | 不进入本轮验证 | A2/910系列路线不能据此替换310P目标；a2实验硬件为310P3/300I DUO，a1为只读生产基线 |

#17640当前API返回mergeable=true/unstable，不能沿用“有冲突”的旧判断。#18057人工APPROVED对应旧提交111fd6c0，当前head已变化；不能宣称当前提交已获批准。#17604当前Open、未合并、mergeable=true/unstable。上述状态不代表检查全部通过。

## 依赖边界与未核实事项

#18057固定ops-transformer `0340eae051ddf5ab956df0f7e0396dbd6314cfc5`，需要匹配的算子run包及Python wheel；Docker基线CANN9.2.0_20261022115200。当前生产CANN9.1.0-beta.1/torch-npu2.10.post2，不能从PR存在推导已安装算子可用。

补充报告所述ops-transformer MR !11242 WY、!12078 FusedGdnGating“已合并master”尚未获得可读官方内容核实，保持待核实；即使master已合并，也不代表生产发行包已包含。未取得固定外部commit全部算法源文件，不能宣称与#17640底层算法等价。gating输入转FP16可能改变误差，应与WY分开验收。当前没有升级CANN、驱动或生产镜像。

#17604作者报告prepare局部减少launch与耗时；这些是作者环境数据，未证明本机TP2端到端Decode收益。其新增测试使用约1e-2容差、equal_nan=True，不替代本轮有限值检查和既定严格门禁。当前生产模型源码已将qkv/z合为qkvz、b/a合为ba；不能再将这一既有融合当作新增优化。

## 已完成证据与验收边界

完整配套H/O绑定保持固定PR函数体，独立namespace/vendor，不修改生产镜像。两卡、Torch/native WY四组各48次，192/192输出在既定误差内且重复bitwise稳定；两个新进程各10/10严格用例，包括独立CPU参考的输出/末状态、非零state、变长与组头比例。192次链路探针本身未覆盖每次末状态，不能写成192次全状态验证。算子通过不等于整模型验收；也未证明完整根因。

35B W8A8 TP2候选前后短请求各8/8、事实16/16、业务自动检查28/28、GSM64为63/64。人工会议4负责人错配且将计划写成完成，最终 **not_accepted_business_semantics**，未运行性能。GSM与原始生产冻结基线64个索引和预测答案一致，文本仅19/64完全相同；不构成逐token/logits等价。生产A′也曾出现负责人错误，不能宣称候选独有回归或错误由GDN单独引起。

原始WY-only Prefill +38%～54%仍仅属于此前C1零命中TP2实验；完整H/O+WY此次没有性能数值。数据见 `results/upstream-gdn-pair-v2-20261104/`、`results/production-gdn-pair-quality-20261104/`。

## 下一步顺序

1. P0保留独立Torch参考及上游配套H/O诊断，分开记录bitwise、数值容差、答案、语义；先补负责人/任务/日期关系的可复现基线与候选对照，必要时再定位真实输入和logits，避免凭GSM通过宣布修复。长上下文、更多chunk边界、mixed state和greedy覆盖仍不完整。
2. P1先归因生产Decode的实际调用/shape，再选小范围布局、调度或预处理优化。#17604为参考，310P执行兼容和质量需独立证明；不同时堆叠WY、gating和Decode改动。
3. P2继续把Dense507015与GDN分开。无权重合成复现已具备反馈价值，但未定位为vLLM-Ascend Python缺陷；post4镜像虽可启动，短请求7/8未过，仍无可验收Dense性能基线。

上游贡献优先最小复现、生产形状测试和有根因的窄修复。未提交独立重复WY PR，未对外发布issue/comment，也未合并到私有远端。

## 后续CANN版本调查补充

已进一步核实正式v0.30.0完整配套使用CANN9.2.1/post7，固定310P Dockerfile存在。建议把新版默认完整栈作为a2下一阶段优先对照，先于移植#18057。该310P Dockerfile显式卸载Triton，#17604仍仅参考。详见cann-upgrade-evaluation.md；“9.2需验证”不代表只能选择PR当时9.2.0环境。
