# GDN 上游候选评估与当前取舍

日期：2026-11-04。用户补充研究作为线索，采用“公开代码审计、隔离验证、选择性采用、增量贡献”的路线。当前继续固定版本验证，不将所有候选依次移植。官方GitHub API元数据、patch与review快照保存于 `results/upstream-gdn-pr-audit-20261104/`；状态只代表查询时刻。

## 当前决策

| 候选 | 决策 | 依据与最小后续范围 |
|---|---|---|
| #17640，head `18163f6b1d208ab3b4fb06e9752225611c37f40e` | 保留隔离诊断；暂不采用 | 数值覆盖见[Profiling](../ascend310p-profiling.md)，模型判定见[结果](../ascend310p-results.md)。先贡献生产形状回归及最小复现，停止扩大独立WY实现 |
| #18057，head `2620e77d7bdb3c54c6ea98b24b570269f6eed6b7` | 依赖核对后备选 | 接入外部CANN WY/gating，未替换完整H/O。生产9.1-beta，PR Docker为9.2.0；先核对匹配的run/wheel、依赖和API，再决定单独WY验证，gating另作变量 |
| #18410，head `988fa61e81b9…` | 备选，当前不叠加 | 初始化时绑定算子，与#18057 dispatch路线择一，避免两套加载/回退同时进入生产 |
| #11941 / #13122 / #17412 | 参考，不重复实现 | 原生WY及CANN集成已有重叠；保存代码差异，只有独立缺陷/支持范围/收益证据才考虑独立PR |
| #17604，head `630e2729d7e2e0476373e8c72010015fec940a33` | P1参考；310P兼容未证明 | 融合QKV整理、L2Norm与gating的Triton prepare。改动通用ops/triton和运行时patch，未提供本轮310P生产栈执行证据；参考310P gating函数不等于支持310P部署。暂不直接移植或引用作者Decode收益为本机收益 |
| #8872 MegaGDN | 不进入本轮验证 | A2/910系列路线不能据此替换310P目标；a2实验硬件为310P3/300I DUO，a1为只读生产基线 |

#17640当前API返回mergeable=true/unstable，不能沿用“有冲突”的旧判断。#18057人工APPROVED对应旧提交111fd6c0，当前head已变化；不能宣称当前提交已获批准。#17604当前Open、未合并、mergeable=true/unstable。上述状态不代表检查全部通过。

## 依赖边界与未核实事项

#18057固定ops-transformer `0340eae051ddf5ab956df0f7e0396dbd6314cfc5`，需要匹配的算子run包及Python wheel；Docker基线CANN9.2.0_20261022115200。当前生产CANN9.1.0-beta.1/torch-npu2.10.post2，不能从PR存在推导已安装算子可用。

补充报告所述ops-transformer MR !11242 WY、!12078 FusedGdnGating“已合并master”尚未获得可读官方内容核实，保持待核实；即使master已合并，也不代表生产发行包已包含。未取得固定外部commit全部算法源文件，不能宣称与#17640底层算法等价。gating输入转FP16可能改变误差，应与WY分开验收。CANN容器实测见[完整栈评估](cann-upgrade-evaluation.md)。

#17604作者报告prepare局部减少launch与耗时；这些是作者环境数据，未证明本机TP2端到端Decode收益。其新增测试使用约1e-2容差、equal_nan=True，不替代本轮有限值检查和既定严格门禁。当前生产模型源码已将qkv/z合为qkvz、b/a合为ba；不能再将这一既有融合当作新增优化。

## H/O接口契约与贡献范围

#17640配套H/O须连同原binding的NZ初始状态约定验证；新O/旧H混合组合失败不能外推为完整PR错误。独立namespace/vendor，并在bootstrap后、dlopen前校验加载顺序和实际映射库。固定head提取函数体，保留原作者归属。

算子检查范围、独立CPU参考及逐位重复证据见[Profiling](../ascend310p-profiling.md)；9.2.1重新构建、API同名绑定问题和扩大门禁见[CANN实测](cann-upgrade-evaluation.md)。候选吞吐与业务判定统一见[结果](../ascend310p-results.md)。

贡献优先最小复现、生产形状/非零状态/重复性测试及有根因的窄修复，不另提重复WY功能PR。#18730是此前ragged MTP状态行修复，不重复计入本轮。后续执行顺序见[计划](../ascend310p-optimization-plan.md)。
