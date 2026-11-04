# Decode 候选只读审计（2026-11-04）

本轮P0模型质量运行期间，只读取权重量化描述及既有Profile，没有并行发送NPU负载。

生产W8A8的量化描述明确：40层MoE路由专家的gate/up/down均为W8A8_DYNAMIC；30层linear attention的in_proj_qkv/z/b/a和out_proj均为FLOAT；40层shared_expert的gate/up/down为FLOAT；10层full attention的q/k/v为W8A8_DYNAMIC但o_proj为FLOAT；lm_head也为FLOAT。FLOAT表示checkpoint量化方法标记，不独立证明运行中每个Matmul的dtype/耗时归属。

既有W8 TP2 Decode Profile：MoE grouped matmul约23%、主要FP16 NZ MatMul变体约20%、PA mask约12%、lm_head对应形状约9%。多数Graph内投影shape=N/A，不能凭量化描述就精确归因20%的每一层。

下一步P1应先做最小范围调用/形状归因，再选择保持语义的投影/布局/调度优化。lm_head或投影INT8会改变量化误差，需要独立质量门禁，不能作为默认低风险动作；也不能把已有W8A8专家再次“量化”为新贡献。当前没有新增Decode性能收益。

## 运行时代码修正

冻结的vLLM源码已把checkpoint的in_proj_qkv和in_proj_z合并为in_proj_qkvz，把b/a合并为in_proj_ba；无LoRA默认路径创建这两个投影。不能把“qkv与z从分离到融合”作为新增优化。量化描述中的五个checkpoint字段并非五次运行时Matmul。投影形状和Graph内耗时仍需实际归因；本轮未执行新NPU微基准。证据见projection-init-matches.json及projection-source-matches.json。
