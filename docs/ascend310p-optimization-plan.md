# Ascend 310P 优化优先级与采用条件

更新于2026-11-04。生产配置身份与测量协议见[基线](ascend310p-baseline.md)，候选比较和最终判定见[结果](ascend310p-results.md)。

## 当前优先级

评估按负载画像分别报告，不只看会议回放：会议纪要（Decode 为主）、知识库/RAG（8K–32K 输入短输出，Prefill 为主）、通用基准（1K/4K/16K/32K × C1/C3/C4）。

1. **WY 收口。** 9.2.1 配套 H/O 下只切 Torch/native WY，三臂 A/B/A′ 过模型级数值门禁（logprob 偏差不超过 A′ 噪声底）和任务非劣门禁，再测性能。采用看 Prefill 基准与 RAG 收益，会议回放非劣即可。上游只为现有 PR 补测试，契约见[上游评估](research/gdn-upstream-pr-evaluation.md)。
2. **Decode。** 会议回放约 89% 耗时在 Decode。先补测带阶段标签的回放和多并发 Profile，拆清 FP16 矩阵乘（共享专家/投影）归属；再优先做可逐位一致的冗余 TransData 和中等成本的 lm_head，大成本 kernel 项依补测结果定。
3. **27B 基线。** 门槛改为相对同轮对照非劣，直接走数值与任务门禁。
4. **预算档位。** 35B W4A8 TP1，再 27B W8A8 TP1 容量筛选、27B W4A8 适配。

自由文本业务语义（负责人、完成状态等）在所有栈和 GPU 上都存在，属于上层提示词/协议问题，不阻塞本线。

## 采用条件

- 候选只要求相对同轮对照非劣：算子对 CPU 参考，模型级 logprob 对噪声底，任务（GSM64、结构化字段、短题）通过数差 ≤1。人工业务审阅只在采用前做一次作参考。
- 长上下文、chunk边界、非零/mixed state、greedy及状态复用按实际覆盖记录；容量准入不等于满长请求通过。未验证的部署组合不生成上线profile。
- CANN使用固定完整配套栈，构建清理旧产物并验证实际映射库；采用条件与实测见[CANN评估](research/cann-upgrade-evaluation.md)。不叠加WY、gating和Decode改动。
- Prefix Cache保持生产策略，冷/热分别报告；不扩大重复缓存扫描。GDN FP16已有实现，不重复包装为新增优化。MTP、proposer及GPU优化参考须先有本机瓶颈和兼容证据。
- 后续实验保持a1生产只读，a2隔离并在结束后独立检查服务恢复。
