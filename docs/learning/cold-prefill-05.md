# 拆测 310P 分页 Attention：历史读取与外部 mask

2026-11-14，Atlas 300I DUO `ychy-a2`，原验收镜像 `bb0fa66f…`。只读拆解[上一轮](cold-prefill-04.md)的 2048 档双 rank 冷 32K 时间线，保留新 token 数固定为 1920 的 15 个有历史段；另在 device0 持锁跑隔离算子，对照同一 query、同一 FP16 NZ KV 缓存、预先写入的 KV，仅改变历史长度。未启动模型服务，未操作 a1。

实际镜像这条路径没有压缩 mask 接口，走 legacy `_npu_paged_attention_splitfuse`。`forward_chunked_prefill_310` 在 kernel 调用前让 `get_splitfuse_mask` 从 q/context 长度生成 `[Q,C]` 的因果 mask，再转 FRACTAL_NZ。已交付的 `graph_safe_splitfuse` 捕获/重放保护是另一路逻辑；eager 回落仍调用原函数。不能根据文件里有 v2 分支就认为当前镜像走压缩 mask。

| 总上下文 / 历史 token | 分页调用，复用已建 mask | 原路径生成并转换 mask | 仅 ND→NZ 布局探针 | KV 写入 1920 token |
|---|---:|---:|---:|---:|
| 3840 / 1920 | 7.38 ms | 2.83 ms | 0.87 ms | 0.10 ms |
| 7680 / 5760 | 14.26 ms | 5.31 ms | 1.80 ms | 0.10 ms |
| 15360 / 13440 | 27.99 ms | 9.99 ms | 3.43 ms | 0.17 ms |
| 23040 / 21120 | 41.76 ms | 14.70 ms | 5.07 ms | 0.15 ms |
| 30720 / 28800 | 55.52 ms | 19.43 ms | 6.71 ms | 0.16 ms |

固定 Q=1920、每 rank 8 个 query head / 1 个 KV head、head dimension 256、block=64。逐阶段同步的宿主墙钟（两次反序遍历，各 3 次预热、7 次计时）。各阶段独立测量，不能当成设备关键路径求和。五个长度用 CPU FP32 因果参考检查每档五个 query 位置和所有 head，最大绝对误差 `9.88e-5`；独立生成 ND、再格式转换的探针 mask 与原函数得到的整块 FP16 Attention 输出逐位相同。第一次探针因导入服务包触发循环依赖，未执行算子；接续直接从锁定镜像提取原函数体，未改算术。

整模型时间线支持同样的拆分：每 rank、每个匹配段有 11 次分页和 11 次匹配 shape 的比较、选择、转置以及张量搬运。rank0 的分页 kernel 单次中位 **6.94→55.10 ms**；对应 mask 比较、选择、转置的单次中位之和约 **2.10→15.75 ms**，另有同 shape 的搬运 **0.19→1.42 ms**（只按 shape 匹配）。15 段内这些外部 mask 活动合计约 1.64 秒/rank，分页本身约 5.12 秒/rank；两 rank 同向。KV 写入约 0.062 ms/调用，未观察到随历史增长的主要写入成本。复用现成 mask 后，分页 kernel 仍增长约 7.5×；尚不能把这部分只归给 KV 带宽，还包含历史注意力计算及 kernel 内部的 mask 读取等。

下一步的最小候选是 **只在一次 eager Prefill forward 内复用同一份因果 mask**。候选必须按完整 query/context 长度序列、设备和布局区分，forward 结束后释放，不做跨请求缓存，不改现有 Graph 捕获/重放缓冲。先检查不同长度、混合序列、prefix hit/冷请求与 Graph 回归，再用同镜像对照 32K 和固定到达回放。上述 1.64 秒是已观测的算子时间池，不能当作已获得的加速；没有改变部署默认值。

[结构化结果](../../results/prefill-attention-20261114/summary.json)。隔离算子的原始脚本和日志保留在 a2 的 `~/ascend-lab-runtime/prefill-attention-r2-20261114`，本机副本位于 `artifacts/prefill-attention-20261114/`；退出码 0，独立 SSH 确认设备和锁已释放。
