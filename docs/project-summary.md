# 项目技术总结与简历成果描述

数据均来自本仓库 `results/` 下的原始记录；均为固定协议单流测量，不是 P95 或完整精度评测。

## 技术总结

在 Atlas 300I DUO（Ascend 310P3）上适配 Qwen3.6-35B-A3B 与 Qwen3.8-Flash-Next 两个 MoE 模型，主要工作：

1. **W4A8 MoE 量化与算子**：ModelSlim SSZ 对称 per-channel INT4 专家权重，设备端 per-token INT8 激活，以 INT4 limbs 组合 Cube 整数计算；基于社区 310P W4 实现补充 Qwen3.6 加载、TP 专家所有权与 BF16 MTP；发布自包含权重（ModelScope，64 文件 SHA256 全部校验）。
2. **推测解码与 Graph**：MTP1/MTP2 + Decode Graph 在 TP1 上通过 4K/32K/112K 启动与长检索；32K 输入同协议下 MTP2 相对 MTP1 Decode 约 +24.8%（23.20 → 28.94 tok/s）。MTP2 在其他两台机器上未通过，已回退 MTP1，未宣称通用。
3. **长 Prefill 优化**：MindIEQ 分流 + 自有 A8 量化，相对原 W4 的 Prefill 中位提升 8K +23.29%、32K +20.69%。
4. **C8（INT8 KV）**：独立分页 KV 算子，TP1/MTP1/MTP2/Graph 已验证，仍为实验后端，容量收益需计入 FP16 工作区。
5. **上游缺陷定位**：复现并定位 310P GDN 在 MTP 下状态行被覆盖的问题（vllm-ascend issue #17856），提交修复 PR #18730（审阅中）。
6. **Qwen3.8 双 310P（TP2/EP2）**：做容量审计，打通完整检查点服务；定位并修复社区 GDN Prefill 因果掩码错误；Decode Graph 在 512/128 上 13.87 → 16.67 tok/s（约 +20%）；实测到 16K，32K 请求无法调度，已如实记录。

## 简历成果描述（可直接使用）

**中文**

- 在 Ascend 310P3 上完成 Qwen3.6-35B-A3B 的 W4A8 + MTP 推理适配，开源自包含量化权重（ModelScope）和可复现工程；TP1 下 112K 上下文通过启动与长检索验证，MTP2 相对 MTP1 Decode 提升约 25%，MindIEQ 分流使 8K/32K Prefill 提升约 21–23%。
- 定位 vLLM-Ascend 310P GDN 在 MTP 下的状态行缺陷并向上游提交修复（PR #18730）；在 Qwen3.8 TP2 适配中发现并修复 GDN Prefill 因果掩码错误。
- 将 Qwen3.8-Flash-Next 跑通在双 310P（TP2/EP2）上，Decode Graph 提升约 20%，实测可用上下文 16K；以原始数据区分已验证与未验证范围（MTP、C8、并发、质量评估尚未验收）。

**English**

- Adapted Qwen3.6-35B-A3B (W4A8 + MTP) to Ascend 310P3 and released self-contained quantized weights with a reproducible build; TP1 passes 112K-context startup and retrieval checks, MTP2 improves decode ~25% over MTP1, and MindIEQ routing improves 8K/32K prefill by ~21–23%.
- Diagnosed a vLLM-Ascend 310P GDN state-row bug under MTP and submitted an upstream fix (PR #18730); found and fixed a causal-mask error in the GDN prefill kernel during Qwen3.8 TP2 bring-up.
- Brought Qwen3.8-Flash-Next up on two 310P (TP2/EP2) with ~20% decode gain from graph capture and verified 16K context, with validated and unvalidated scope separated in raw data.

## 使用提醒

- 简历中可写「提交上游 PR」，不要写「已合并」（#18730 目前仍为审阅状态）。
- 「社区基础实现」（matteius/opensensor 的 310P W4）与 GDN WY 重写（PR #17640）不是本人原创，措辞上用「基于」「适配」。
- Qwen3.8 当前只证明到 TP2/MTP0/16K。
