# SplitFuse mask 复用：混合生成质量回归

2026-11-14，Atlas 300I DUO `ychy-a2`，沿用[性能对照](cold-prefill-06.md)的同一镜像和模型制品、TP2、batch 2048、Prefix Cache、MTP2 和 Decode Graph。四路冻结输入及到达偏移保持一致，同一个服务进程按 `fresh → reuse → fresh → reuse` 执行；回放取真实 token ID、选中 token 的 logprob 和返回的 top logprob。每轮独立 salt，暖两路分别命中 11520 token，冷两路为 0。

前四轮的 31 个已记录 eager 调度步，按 worker 内请求次序和各请求的 query token 数逐步一致。记录覆盖合计 query 数至少 32 的 eager 步；它不包含全部解码步，也没有固定每步的真实 token ID、MTP 接受数或 KV 内容。

第 2 次 `fresh` 和两次 `reuse` 的四路生成 token ID、每个 token 的选中项及返回的 top logprob 完全一致。第 1 次 `fresh` 与后三轮相比，三路在以下位置首次发生 token 分歧，另一路完全相同。分歧前，比较的是共同生成历史。

| 请求 | 首处分歧的输出 token（从 1 数） | 第一次 fresh → 后三轮的 token ID | 第一次 fresh 中两项的 logprob 差 |
|---|---:|---:|---:|
| 暖跟进 | 188 | 109542 → 126492 | 0.296875 |
| 冷长输入 | 87 | 98293 → 99586 | 0.015625 |
| 冷短输入 | 无 | — | — |
| 暖提取 | 179 | 97990 → 97646 | 0.453125 |

第 1 次和第 2 次的 `fresh` 在同一组位置、以相同的 token 和 logprob 复现了分歧。**观察到的分歧无需启用 mask 复用即可出现。** 记录只能将问题缩小到该镜像/服务回放已有的差异，尚不能定位到某个未记录的 decode 步或 MTP 行为。

独立服务审计在每个实际候选命中处立即拿新建和复用的 mask 跑同一个 legacy SplitFuse 调用。两次使用同一份 query、KV、页表、长度和输出区，保留第一次输出并在 CPU 按 FP16 位模式比较。两个 TP rank 各有 **270 次比较全部一致**，其中各 **180 次**是多请求 batch。每 rank 覆盖 29 组有序长度、最多四路并行、最大 context 32774。重跑审计的四路输出 token 和返回的 logprob 也与第 1 次 `fresh` 完全一致。

首个服务窗口的 HTTP 四轮均已完成，随后审计探针在 forward 结束时检测到没有截获底层算子，故窗口退出 1。将算子拦截放到实际 eager forward 内，单独重跑审计窗口，退出 0，两次均保留原始日志。第二个窗口也通过 Graph 捕获/重放、前后短 JSON 和实际 prefix hit 验收。独立检查确认 a2 无实验容器和 NPU 进程，端口和锁均已释放；a1 未操作。

这通过了本次覆盖范围内的质量门禁，前一轮 32K 延迟和混合吞吐收益可保留。候选适合进入带开关的业务灰度；默认部署仍不变。现有证据没有覆盖真实会议请求、极长上下文或生产尾延迟，也没有定位原路径跨轮差异的底层原因。

[结构化结果](../../results/prefill-mask-quality-20261114/summary.json)；原始目录在 a2 的 `~/ascend-lab-runtime/prefill-mask-quality-20261114` 和 `~/ascend-lab-runtime/prefill-mask-quality-audit-20261114`；本机副本在 `artifacts/prefill-mask-quality-20261114/`。

后续已将通过本轮门禁的同一份源码封装，完成 a2 冒烟并发布为[内部双卡默认交付](../qwen36/w8a8-wy-delivery.md)。新部署默认开启复用；可用 `QWEN36_EAGER_MASK_REUSE=0` 回退。
