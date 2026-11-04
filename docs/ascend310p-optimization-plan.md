# Ascend 310P 优化优先级与采用条件

更新于2026-11-04。生产配置身份与测量协议见[基线](ascend310p-baseline.md)，候选比较和最终判定见[结果](ascend310p-results.md)。

## 当前优先级

1. **P0：35B同权重数值验收。** 在固定9.2.1配套H/O下，只切Torch/native WY，做固定token前缀对照和真实模型输入回放；保留独立CPU参考、有限值、输出/末state与逐位重复检查。合成门禁进展见[Profiling](ascend310p-profiling.md)。
2. **P0：WY性能与上游增量。** 对应数值门禁通过后测隔离算子性能；整模型性能需先通过模型数值门禁。沿用同配置、冷缓存和A/B/A′协议，不迁移旧栈收益。优先为现有PR补充测试或有根因的窄修复，候选契约见[上游评估](research/gdn-upstream-pr-evaluation.md)。
3. **P1：27B与业务协议。** 做同模型数值定位，受控复核期限/状态差异，分别评价原始prompt和证据约束协议。字段/长上下文细节见[配对验证](research/action-state-paired-validation.md)，GPU行为与部署档位见[模型质量研究](research/model-quality-and-deployment-profiles.md)。
4. **P2：Decode与预算档位。** 按[算子证据](ascend310p-profiling.md)先归因MoE、投影/输出头、PA mask及Graph内通信，再选单变量改动。预算档位先35B W4A8 TP1，再27B W8A8 TP1容量筛选、27B W4A8适配；TP2 W4和35B W8 TP1优先级靠后。

## 采用条件

- 算子数值、相对业务回归、绝对业务可用性分别验收。隔离算子性能不等待共享自由文本错误修复；生产采用仍须独立业务验收。schema/事实通过不能替代负责人、期限和完成状态审阅。
- 长上下文、chunk边界、非零/mixed state、greedy及状态复用按实际覆盖记录；容量准入不等于满长请求通过。未验证的部署组合不生成上线profile。
- CANN使用固定完整配套栈，构建清理旧产物并验证实际映射库；采用条件与实测见[CANN评估](research/cann-upgrade-evaluation.md)。不叠加WY、gating和Decode改动。
- Prefix Cache保持生产策略，冷/热分别报告；不扩大重复缓存扫描。GDN FP16已有实现，不重复包装为新增优化。MTP、proposer及GPU优化参考须先有本机瓶颈和兼容证据。
- 后续实验保持a1生产只读，a2隔离并在结束后独立检查服务恢复。本文只记录计划，不授权新的设备操作。
