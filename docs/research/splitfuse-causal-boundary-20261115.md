# 310P compressed SplitFuse 因果边界定位

2026-11-15，a2 双 310P3，CANN 9.2.1 / torch-npu 2.10.0.post7，以镜像 `b217eec3…` 保留的 vllm-ascend `76ea9d24…` 为实验环境。只运行 torch-npu 算子，不加载模型，不导入 vLLM，不用 WY 或 Graph。沿用[上一轮交付结论](wy-compressed-attention-20261115.md)。

## 直接测了哪些未来位置可见

`probe_cutoff.py` 固定 Q=K=0，把 query 对应的 K/V 后缀分批编码：每个待测 key 的 V 独占一个输出通道，幅度为 64，余下为零。对每个 query row，以 `32/context` 为可见阈值；正常可见的一次贡献至少为 `64/context`。因此直接读取可见集合，区分连续越界与带空洞的越界，不从 FP16 总和反推边界。每个样本 8 个 query heads 一致，双卡结果一致。legacy 选择相同输入做对照。

这里 `history = context_len - query_len`，即当前 query chunk 之前的历史 token 数。`+n` 表示相对该行正确因果截止位置往后的第 n 个 key。

| context | query | history % 64 | 受影响 query rows | row 123 实际多见的 key |
| ---: | ---: | ---: | ---: | --- |
| 2048 | 384 | 0 | 0/384 | 无 |
| 2049 | 384 | 1 | 47/384 | +4 至 +19，+1 至 +3 仍被屏蔽 |
| 32768 | 383 | 1 | 40/383 | +4 至 +19 |
| 32768 | 384 | 0 | 0/384 | 无 |
| 32774 | 384 | 6 | 47/384 | +1 至 +14 |
| 32784 | 16 | 0 | 0/16 | 不适用 |
| 3968 | 1920 | 0 | 0/1920 | 无 |
| 3969 | 1920 | 1 | 239/1920 | +4 至 +19 |
| 32774 | 1920 | 6 | 239/1920 | +1 至 +14 |

- 本轮两个网格去重后共 341 种形状（224 种复现），在物理 block=64、head_dim=256 路径上只在 `context > 2048` 且 `history % 64 != 0` 的已测形状中观察到泄漏。对齐总长既不必要也不充分：同一个 32768 可失败，32784 可通过。短 query 的部分非对齐形状保持正确，所以两个条件不是已证明的充分条件。`context=2050, query=3` 也有 2/3 行受影响，不宜假定该问题只影响长 Prefill。
- 最短已复现 context 为 2049。`context=2049, query=1024` 的历史只有 1025，故不是必须有 2048 个历史 token。2048 恰好也是固定 ND mask 的维度；测到的是输入边界，不能仅据此断言内部切换分支的判据。
- 典型 `32774/384` 在完整的 64 行区间末尾反复出现异常，例如 rows 56–63、120–127。row 56 放行 +2 至 +17，却仍屏蔽 +1；row 123 放行 +1 至 +14。两个网格中没有观察到应可见的 query 后缀 key 被屏蔽。带空洞的放行不能用统一对角线平移解释。

## 物理页编号、mask 幅度和几何形状对照

- 将物理页编号反转并同步改写 KV slots，同时把压缩 mask 的屏蔽值从 −10000 改成 −65504。在 block=64/head_dim=256 下，9 组对照的逐行结果没有变化。连续物理页编号和 mask 幅度均解释不了关键反例。
- block=128/head_dim=128 的 `32774/384` 仍有 23 行泄漏；典型异常窗口在 rows 120–127、248–255，较 64 页测试随页几何变化。固定 head_dim=128、使用 block=64 又出现更广的放行，甚至 `32768/384` 历史整页对齐例也有 180/384 行受影响；这个变体在正向页和通常 mask 值下结果也相同。这些 shape 变体说明主实验的对齐观察不能作为通用安全开关。

## 封装源码所能确认的范围

核对 Ascend/op-plugin 固定提交 `9f8981728bd4f7a257c14523266b0bbce1c831e1` 的 `op_plugin/ops/atb/PagedAttentionSplitfuse{,V2}.cpp`。两个入口都把 `context_lens` 和 `seq_len` 直接交给同一个 `PagedAttentionOperation`，均指定 `CALC_TYPE_SPEC`；关键区别是 legacy 用 `MASK_TYPE_SPEC`，v2 用 `MASK_TYPE_NORM_COMPRESS(5)`。公开封装未做 history 取整，未检查历史对齐，也未解释 2048 边界。这是公开源码对照，未证明该提交与已安装 wheel 的构建源码完全相同。

同一提交的 `test/test_custom_ops/test_atb_paged_attention_splitfuse_v2.py` 使用 310P、head_dim=128、block=128，prefill context 固定为 1024，历史前缀为 128 的倍数。它检查的固定 ND mask 和 `context_lens`/`seq_len` 语义与本复现一致，但这组测试不覆盖本次的超 2048、非整页历史。

**目前最有依据的定位是 ATB PagedAttention 的压缩 mask 路径，在长上下文下组合非整页历史与局部因果 mask 时，部分行错误放行了未来 key。** 64 行周期和可见集合中的空洞更符合局部分块映射/合并错误，而非 FP16 舍入、vLLM 长度传错或整张三角 mask 统一平移。未取得对应 ATB 实现源码，无法判断具体是 host tiling、设备端 mask 地址，还是 tile 拼接分支。需要维护者对输入约束和内部实现作进一步确认。

本轮不修改路由、不替换正式镜像。当前观察到的历史对齐条件仅用于进一步复现和定位，不是经过验收的生产 fallback 规则；若要采用分流，应先覆盖多请求、Prefix Cache、MTP Graph 和真实调度形状，以及正确性与性能门禁。

[机器可读证据](../../results/wy-compressed-20261115/causal-boundary.json)包含选定行、双卡比较、原始日志摘要和 SHA256。
