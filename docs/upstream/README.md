# 310P legacy SplitFuse 上游回馈

> 状态（2026-11-14）：legacy 分配补丁归档，暂停投稿。内部默认继续采用已验收的 WY 镜像；当前优化主线是该镜像的 eager forward mask 复用。本文保留为调查记录，不是待发布计划。

2026-11-14 核对 `vllm-project/vllm-ascend` 的 `main`，基线为 `76ea9d24f155afe032ad9ac6c98717f1f7334f36`，其 `.github/vllm-main-verified.commit` 指向 `24420b7a341af37a9bc3186e6fc6062512e0aa58`；310P 服务对照采用同一源码的 Dockerfile 默认 vLLM 配对 `v0.33.0`。基线的 legacy SplitFuse 先构造 `max_model_len²` 常驻 FP16 mask 再取行；生产验收镜像已经使用活跃矩形。因此上游补丁与[已测跨层复用](../learning/cold-prefill-06.md)拆开，收益分别验证。

第一项[最小补丁](../../patches/upstream/310p-legacy-splitfuse-active-mask.patch)只改 legacy eager chunked prefill / prefix hit 的构造宽度。压缩 mask、SpecDecoding、Graph 捕获和初始 FlashAttention 沿用原路径。用上游原和候选 mask 类配合 a2 实际 legacy 算子运行，三组固定宽度对照整块 FP16 逐位一致；另测到 32774 context，CPU FP32 样本误差 `1.54e-4`，长例不分配全宽参考。在无 WY overlay 的官方 `v0.30.0-310p` / CANN 9.2.1 镜像上重复调用 legacy 算子，得到同样的数值结果。单独测 mask 分配：上限 12,288、活跃宽 8,192、query 2,048 时，移除了每 worker 的 288 MiB 常驻方阵，打包后 48→32 MiB。补丁通过 `git apply --check`、Python 语法检查和三个隔离 CPU 测试；纯 `main` 服务的默认基线已完成；强制 legacy 的补丁前后对照已完成，混合文本差异已在原版重复复现；强制 legacy Graph/MTP 被原版捕获问题阻塞，eager MTP 单独测。原始运行在 a2 `~/ascend-lab-runtime/upstream-splitfuse-20261114`，本机副本 `artifacts/upstream-splitfuse-20261114/npu/`，设备已独立清场。[结构化结果](../../results/upstream-splitfuse-20261114/summary.json)。

准备了英文 [issue 草稿](310p-legacy-splitfuse-issue.md) 和 [PR 草稿](310p-legacy-splitfuse-pr.md)，尚未发布。相关讨论可参考 #11907（压缩 mask 能力）、#16052（MTP Graph）、#18390（已合并的另一个 310P 性能修复）。首个 PR 送审前，用 310P Dockerfile 默认的 vLLM 与 CANN 组合跑无缓存首段、切片、prefix hit、混合和 Graph/spec 回归，记录纯引擎的延迟、mask 分配及峰值内存。初始 FlashAttention 仍有独立的大 mask 约束，不能把这个补丁宣传成上游已支持 262K。

第二项是同一 target forward 的跨层复用。优先让共享的 attention metadata 持有结果并明确生命周期；先验证 MRv1/MRv2 是否都有这个语义，避免引入模型特判、进程级缓存以及每层新增 D2H。对异步 MTP 使用设备校正后的长度；至少覆盖冷输入、prefix hit、混合与 Graph 路径，单独用开关测量首 token 延迟及 NPU 峰值内存。

[main A/B 对照协议](310p-main-ab-plan.md)明确区分官方 main、只加矩形 mask 的候选、以及本地集成 #17640 的历史验收镜像。新版运行时可能实际走压缩 mask；只有确认命中 legacy 的样本才能衡量补丁收益。

纯 `main` 同镜像服务基线：两 rank 默认均走压缩 SplitFuse。4K/8K 各 3 次冷请求的 TTFT 中位为 3.36 / 6.75 s，8K 检索通过。四路混合实际 prefix hits 为 `[3840, 0, 0, 3840]`。这组基线没有经过补丁目标路径；强制 legacy 测量单列。

强制 legacy 同镜像 A/B 的两个 rank 首个采样调用宽度从 12288 到 3840；4K/8K 冷 TTFT 中位 3.421→3.399 / 6.901→6.863 s（各臂各档 3 次），整卡采样峰值分别降 395/396 MB。8K 检索均通过，混合缓存命中相同，但四路仅两路生成文本相同。结果不外推至默认压缩路径；原版重复组与候选四路文本全一致，首轮差异可在两次原版之间复现。强制 legacy 的原版 Graph/MTP 捕获因 `query_start_loc.to("cpu")` 触发同步 D2H 失败；候选未跑 Graph，eager MTP 单独对照。
