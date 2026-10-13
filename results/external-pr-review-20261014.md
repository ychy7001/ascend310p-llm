# 外部算子候选复核（2026-10-14）

## 结论

目前不把 MindIEQ `fake_quant_matmul_x2nz_3_r3_biased` 直接并入 Qwen3.6 W4A8 服务。该分支的 `fused_quant_mat_mul` 文档将产品支持列为 Atlas A2 训练/推理系列，并明确将普通 Atlas 推理系列（310P 所属产品线）列为不支持；其 AscendC 实现主要位于 `arch35`，测试配置也以 `ascend910B1` 为主。因此，接口文档里出现 INT4 不能证明 310P 可执行，必须先做 310P 编译和真实设备 smoke test。

这不意味着它没有价值：它提供了一个比当前自定义 Python 侧路径更标准的 fused quant matmul 接口，包含 INT4/INT8 输入、scale、bias，以及 NZ 权重的 API 设计。若 CANN 9.1/310P 实际暴露相应 kernel，可作为后续替换 `QwenW4A8Int4MatmulV310` 的候选；在没有 310P 构建产物、算子注册和数值/延迟结果前，不应称为收益。

## vLLM-Ascend PR #17640

该 PR 的标题和改动是 **310P GDN prefill** 优化，不是 W4A8。它增加 `chunk_gated_delta_rule_compute_wy` 的 AscendC/ACLNN 算子，使用 `arch20/compat_310p.h`，并保留不支持 shape 的 Torch fallback；同时优化 FwdH/FwdO、数值稳定性、NZ state、因果 mask 和若干 Python 调度。PR 描述给出的 standalone Qwen 27B GDN shape 结果为：compute-WY 约 15.1x、完整 GDN 约 8.3x，但这些是独立算子延迟，不能直接换算成 Qwen3.6-35B-A3B 端到端收益。

该 PR 与本项目相关，因为 Qwen3.6 路径包含 GDN/混合注意力时，prefill 可能受同一类 WY/FwdH/FwdO 开销影响。优先级应高于改动 MindIEQ 的 `arch35` W4 算子：先用同一模型、同一容器版本做 GDN standalone shape replay 和小端到端对比；若 PR 分支能在 310P 编译并覆盖实际 shape，再考虑移植其最小提交集。不要把整套 PR 直接 cherry-pick 到生产分支。

## 推荐门禁

1. MindIEQ：在隔离目录构建，目标明确指定 `SOC_VERSION=Ascend310P`；要求注册成功、INT4/INT8 NZ shape smoke、FP16 对照误差和 batch 1/8 延迟均通过。若出现 unsupported arch 或 fallback，停止继续移植。
2. PR #17640：只构建 GDN 相关算子与 Python dispatch，先运行其 310P 单算子回归；再在当前 TP1/48G/C8/MTP2/Graph 服务上测 8K 和 112K prefill/decode。端到端收益至少需要重复三次，且保持输出正确性。
3. 任何候选只有在真实 310P 上有可重复收益后，才整理成主线小提交；否则保留复核记录，不污染默认 W4A8 路径。

当前已完成的 LM head W8A8 独立实验表明：四分片 W8A8 batch=1 eager 约 3.249 ms，预分配输出的 Graph 约 2.998 ms，Graph 相对 eager 输出最大误差为 0；但 FP16 仍只有 ND 参考，尚无公平生产 NZ 基线，因此该结果不构成端到端加速结论。
