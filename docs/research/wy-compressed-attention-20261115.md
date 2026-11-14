# WY 与 compressed Attention：组合可运行，压缩算子因果门禁失败

归档日期：2026-11-15。远端原始日志为机器 UTC 时间 2026-11-14；不将日志时钟误作交付日期。当前默认仍为正式发布的 `qwen36-tp2-w8a8-wy-mtp2-20261114`，digest `sha256:75a20b2954edae93e5990f795e12c09c06faadcc49d6acd880cbc2f5f8ab217d`。这是已验收交付，不是本轮未通过的新栈候选。

## WY 不要求 legacy eager

native WY 对应 GDN/linear attention；compressed SplitFuse 对应 full attention，两者没有必然绑定关系。旧交付栈 torch-npu 2.10.0.post2 缺少 `_npu_flash_attention_v3`、`_npu_paged_attention_splitfuse_v2`，因此其能力检测选择 legacy。该检测检查算子是否存在，不检查是否启用 WY。

本次候选使用已有 v0.30/CANN9.2.1/torch-npu2.10.0.post7 镜像 `fdbafc8b…`，同时保留 native WY 与 compressed 分支。与当前正式镜像逐文件核对，WY Python wrapper、扩展 `.so`、custom OPP opapi/proto/tiling SHA 均一致；这证明相应构件相同，不将其等同为新栈整体精度已验收。

## 已完成的服务组合验证

使用同一已验收 W8A8 权重、TP2、FP16、MTP2、async scheduling、chunked prefill、prefix cache、max-num-seqs=4、batch tokens=2048，配置 max-model-len=262144：

| 检查 | 结果 |
| --- | --- |
| 两 rank 实际 native WY 调用 | 通过，一次性调用后日志 |
| 两 rank 实际 compressed SplitFuse 调用 | 通过，一次性调用后日志 |
| Decode Graph 捕获与实际回放 | 单档 `[12]` 通过 |
| MTP2 实际工作 | drafts counter=11 |
| 32K 固定 token 检索 | 通过，返回 `ASCEND732941` |
| 容器退出与清场 | 退出 0，非 OOM，a2 无 NPU 进程 |
| 三档 Graph `[3,6,12]` | 第三档附近事件资源耗尽，capture_end 507903 |

32K 检索耗时约 25.41 秒，包含生成等阶段，仅为功能筛查，不与旧版 TTFT 当作同协议性能比较。KV cache 报告 617467 tokens、理论并发 2.36；未重测完整 262K 容量。本轮未发布新镜像。

## 压缩算子的独立精度缺陷

真实 NPU 对照使用物理 block=64、8 query heads、1 KV head、head_dim=256、FP16 NZ KV cache。模型逻辑 block=640 会拆成十个物理 block，不能直接以 640 调用此算子；此前 640 诊断的 legacy setup 失败作废。

随机输入中，单请求对齐 context 的 legacy/compressed 完整输出逐位一致。未对齐 context 的单请求、混合请求都可出现差异；因此不能仅归因为混合调度。失败例：qLens `[384,1024,512]`、contextLens `[32774,1024,12544]`，row=123、head=6。legacy 对正确 CPU FP32 参考最大样本误差约 `3.92e-5`，compressed 约 `0.00671887`。compressed 与因果截止位置向后移动 14 token 的 CPU 参考误差仅 `1.93e-5`。

进一步用不加载模型、不导入 vLLM、不执行 WY 的最小脚本验证因果不变量：固定 Q=K=0，将 row=123 因果边界后 14 个 V 从 0 改成 64，该行输出必须保持 0。

| context / 运行栈 | legacy row 输出 | compressed row 输出 |
| --- | --- | --- |
| 32768，对齐对照 | 0 → 0 | 0 → 0 |
| 32774，候选新栈 | 0 → 0 | 0 → 0.0275421143 |
| 32774，保留的上游 main 快照 | 0 → 0 | 0 → 0.0275421143 |
| 32774，当前正式交付 | 0 → 0 | 该版本无 v2 算子 |

两张 310P3 均复现。最后一行 query 允许读取这些 V，两种算子输出均约 0.02734375，证明 KV 修改确实生效。该检查不依赖随机误差阈值、业务文本或参考模型生成结果。

上游快照镜像为 `b217eec3…`，标签记录 vllm-ascend revision `76ea9d24f155afe032ad9ac6c98717f1f7334f36`，vLLM source revision `4ba787ec9959e566d3b587a796dff07673324da5`。这是本机保留的固定快照，并非声称当前远端最新 main。本轮复现在该镜像中直接调用 torch-npu 算子；没有重新验证其完整服务。结论定位到该组合下的 compressed 算子行为，尚未定位到 CANN/torch-npu 内部具体代码行。

## 交付与上游反馈

本轮**组合验证完成，升级候选不予晋级**。精度失败后停止性能放行，不通过放宽阈值或关闭 WY 来包装升级。现有已过测 WY 正式默认持续有效，a1 在线服务未操作。

[最小复现及执行说明](../../experimental/wy-compressed/README.md)、[机器可读结论](../../results/wy-compressed-20261115/summary.json)、[上游 issue 草稿](../../experimental/wy-compressed/upstream-issue.md)随源码维护。缺陷修复后，先重新通过相同因果不变量和随机混合数值门禁，再恢复新旧栈固定协议性能、Graph 档位和完整封装验收。当前没有可诚实宣称已验证有效的原生算子修复 PR。
