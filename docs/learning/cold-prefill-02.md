# Qwen3.6 W8A8 接手记录：a2 冷 Prefill 基线

2026-11-14，起点 `91efa51`。a2 使用 Atlas 300I DUO 的两个 Ascend 310P3，驱动 26.1.1。固定验收镜像 `bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4`，对应历史 digest `a11b8aca…`。复用 a2 已有 ModelSlim W8A8 checkpoint，只读挂载；校验配置和分片索引元数据，不宣称重算全部权重分片。实验持 `/tmp/a310-npu.lock`，服务仅监听 a2 回环端口 18147，未操作 a1。

## 哪些优化值得保留

| 路线 | 仓库证据与取舍 |
|---|---|
| native WY + 配套 GDN 内核 | 历史成对冷 Prefill 4K–32K 快约 28–31%；最终组合另行验收，不能把整栈变化都归给 WY。保留与镜像匹配的 OPP、状态表和 slot int64 修复。 |
| 主模型 Graph `[3,6,12]` + draft eager + MTP2 | 同镜像 4K/512 冷对照，MTP0/1/2 的单流服务端 Decode 为 33.10/40.58/43.69 tok/s，C4 聚合端到端为 61.94/71.10/70.04 tok/s。默认沿用 MTP2；四并发长输出占主导时单独验证 MTP1。 |
| 262144 上下文、seq4、prefix cache | 262016+128 已历史通过；seq4 是调度上限，池约 63.5 万 token，放不下四条满长请求。缓存开启经过业务候选验证；冷热计时要分开。 |
| MoE 小 M、QKVZ、CANN 升级 | 小 M 五对会议回放中位只快 1.53%，未达 3% 门槛。QKVZ 量化算子门禁通过，但候选整模型未测；预算不能当实测。CANN 9.2.1 默认 GDN 在扩大样本后仍有数值失败，留在独立分支验证。 |

参考：[制品及受控 Decode 实验](../qwen36/wy-mtp-production-20261108.md)、[WY 完成报告](../qwen36/wy-optimization-completion-report-20261108.md)、[MoE 小 M](../../results/moe-small-m-20261105/README.md)、[QKVZ](../../results/qkvz-w8a8-20261105/README.md)、[CANN](../research/cann-upgrade-evaluation.md)、[业务临时验收与回滚](../../results/wy-mtp-meetai-business-20261109/summary.json)。

## 本轮方法

按[第一课](cold-prefill-01.md)读入口、调度器与 GDN 算子。镜像入口与旧验收 SHA 一致；实验副本只加 `--profiler-config`，关闭 frontend/stack。TP2、seq4、batch2048、MTP2、主 Graph 三档、draft eager、prefix off。固定 token IDs、C1、1 输出，每档预热一次、计时三次，独立 salt 并核对零命中。客户端 TTFT 与服务端 Prefill 阶段吞吐分别记录。采样单独运行，不能把带采样请求的耗时计入基线。

| 冷输入 | 有效次数 | TTFT 中位 | Prefill tok/s 中位 | 相对 11-08 同 token IDs 的吞吐 |
|---|---:|---:|---:|---:|
| 4K | 3/3 | 2.012 s | 2,039.95 | −0.47% |
| 8K | 3/3 | 4.249 s | 1,930.51 | −0.38% |
| 32K | 3/3 | 22.471 s | 1,459.31 | +0.18% |

三档均为 0 prefix hits。启动报告 KV pool 636,903 tokens，满上下文容量系数 2.43×；本轮没有重跑满长或并发容量测试。采样结束后普通 4K 请求通过，TTFT 2.003 秒。实验服务 ExitCode 0、无 OOM，独立 SSH 确认端口已释放、rootless/rootful 均无运行容器、NPU 无进程、锁可获取。

## 4K 双 rank 采样

两 rank 各 8,863 条 kernel 记录，设备事件跨度约 1.99 秒，命名 kernel 区间并集覆盖约 96.7–97.4%。这是采样跨度内的覆盖；差额包含其它设备活动与等待，不能直接归给 Host。Host trace 中带 synchronize 名称的区间并集约 1.77–1.81 秒，与设备计算重叠，不能将其视为可消除的 Host 空耗。

| 算子或组合 | 次数 / rank | 各 rank 累计 kernel 耗时占比 |
|---|---:|---:|
| W8A8 MoE grouped matmul | 160 | 29.2–29.7% |
| MoE 路由、反排与 TopK | 336 | 约 8.7% |
| MatMul5/v2（还未按模块分解） | 470 | 7.0–7.2% |
| native WY + FwdH/FwdO | 各 60 | 9.4–9.8%（WY 单独 6.3–6.7%） |
| 两个主要 TransData 变体 | 各 470 | 4.5–4.6% |
| PagedAttentionDecoderNzMask + UnpadFlashAttentionNzEncoder | 13 + 11 | 6.6–6.7% |

60 次 ComputeWy 确认该请求执行了 native 路径，且调用数与 30 个 GDN 层、两轮计算一致。上表分母是每个 rank 的 kernel 累计耗时，不能相加跨 rank、视为端到端提速，也不能将 4K 构成外推到 32K。原始 profile 包留在 a2；本地保存导出的双 rank CSV、Chrome trace 和普通请求。数值、文件 SHA、实际 PID1 参数统一记录在[结构化结果](../../results/prefill-takeover-20261114/summary.json)。

## 下一步只改一个变量

1. **先试调度预算 2048 → 4096。** 用同一镜像/权重和相同 token IDs 做 A/B/A；优先 4K、32K 冷 C1，再看 C4 混合长短请求的 TTFT、TPOT 和聚合吞吐。4096 可能减少调度轮次和重复的固定工作，也可能增加峰值激活、缩小 KV 池或让短请求等待更久。检查启动、Graph、短 JSON、长检索和实际 KV 容量后再讨论采用。过去 batch4096 的旧栈不构成此处的同制品对照。
2. **跟一条 32K trace。** 对照前后调度块中 Attention 的 KV 长度、MoE 大 M kernel 和布局搬运。如果 MoE 仍占主导，先按 prefill 的 M、专家分布和投影 shape 拆开 grouped matmul 与 routing；不要重做已经裁决的小 M Decode 优化。如果主要新增时间来自历史 KV 读取，再定位 Attention 分页和 mask。
3. **把生成策略和量化独立开。** 用实际长输出和到达分布比较 MTP1/2，记录接受率、TPOT 和聚合吞吐。只有新栈归因仍显示 QKVZ 要紧，再跑现有候选的 A/A′/B 数值与业务门禁。
