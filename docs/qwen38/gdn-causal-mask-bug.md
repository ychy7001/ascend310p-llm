# 社区 GDN Prefill 因果掩码错误：从 token 56 到向量化修复

Qwen3.8 TP2 适配中，一个只在 Prefill 输出上暴露的异常，最终被缩减为 `chunk_fwd_o_vllm` 的因果掩码错误。
固定输入重复执行时，错误元素数仍会变化；把掩码直接变成可观测输出，才将问题从 GDN 数值链路收敛到两处掩码生成代码。
本文依据 [2026-10-18 原始记录](../../results/qwen38/tp2-serving-20261018/README.md#本轮发现及修复gdn-prefill-因果掩码错误)与[最终补丁](../../patches/qwen38/gdn-causal-mask-vector.patch)，只讨论这个算子 bug。

## 1. 现象：T≥64 后，token 56 的输出超出误差门限

测试使用 Qwen3.8 TP2 的实际局部几何：`HK=8、HV=24、DK=DV=128`，recurrent state 为 FP32。
设备为 a2 的一张 300I DUO、两颗 310P3；两个进程的 HCCL all-reduce 检查得到预期的 `1+2=3`。

未修改的社区算子包呈现出明确的长度分界：

- Decode `T=1/2/3/4/8` 正常，Prefill `T=32` 正常。
- 已测的 Prefill `T=64/70/128/256` 在 token 索引 **56** 出现超门限输出。
- 两个 rank 都能复现，固定输入重复执行时，错误元素数每次不同。

这里的“越界”是数值误差超过原门限。长度分界和重复执行的不稳定性，使排查重点落在 64-token 路径的实现上。

## 2. 最小复现：让输出直接显示因果掩码

完整 GDN 链路包含 WY 输入变换等环节，直接比较最终输出不容易判断偏差来自哪里。
[独立掩码探针](../../tools/qwen38_chunk_causal_probe.py)直接调用 `chunk_fwd_o_vllm`，构造以下输入：

- Q、K 在同一个坐标取 1，其余坐标取 0，使所有 token 对之间的 `QK=1`，调用时 `scale=1.0`。
- V 的前 64 列为 `64×64` identity basis，其余列为 0。
- 历史状态 `H=0`，门控 `g=0`，消除历史状态贡献，并使指数门控为 1。

于是输出前 64 列应严格等于 `tril(ones(64,64))`：对角线及其下方为 1，上三角为 0；其余输出列应为 0。
该构造无需模型权重，也无需 WY 计算；每个 head 都能直接检查掩码的每个位置。
预期值只有可精确表示的 0/1，因此上三角出现非零值就是直接的因果性反例。

未修改社区 OPP，两个 rank 各运行 3 次，得到：

| Rank | 三次错误元素数 | 错误行 |
|---|---|---|
| 0 | 13 / 27 / 30 | 56 |
| 1 | 21 / 23 / 28 | 56 |

原始逐次输出见 [rank0](../../results/qwen38/tp2-serving-20261018/baseline/chunk-causal-rank0.json)与 [rank1](../../results/qwen38/tp2-serving-20261018/baseline/chunk-causal-rank1.json)。
同一精确输入产生不同错误数量，这些差异已超出正常 FP16 舍入的解释范围。

## 3. 定位：初始化和 epilogue 各有一份掩码生成逻辑

源码中存在两个需要同时处理的位置：

1. `gdn_fwd_o_kernel.hpp` 的 `InitCausalMask()` 初始化 `64×64` 下三角掩码。
2. `block_epilogue_gdn_fwdo_qkmask.hpp` 的 64-token epilogue，在指数门控计算之后再次逐行生成上三角零值。

原初始化先把整个掩码填为 1，再将上三角清零；清零长度至少为 8 时使用 `Duplicate`，不足 8 时逐元素 `SetValue`。
epilogue 也有类似分支：长段使用向量 `Duplicate`，靠近对角线的短段使用 Scalar `SetValue`，之后继续向量乘法。
这解释了为什么只修 `InitCausalMask()` 仍然失败：epilogue 保留了另一条短标量写路径。

“**短标量写与向量流水线交错**”是源码与对照实验支持的故障解释。
实验未做底层指令跟踪，因此这里把它作为有证据支持的实现层解释，具体指令级时序仍未确认。

另一个关键观察是：这些用例的 GDN 状态误差仍然很小，**只检查 final state 会漏掉问题**。
状态更新和输出计算需要各自的数值判据；用能暴露掩码的结构化输入，可以发现随机输入或状态检查不易揭示的错误。

## 4. 修复：统一使用向量生成和消费掩码

[最终补丁 `gdn-causal-mask-vector.patch`](../../patches/qwen38/gdn-causal-mask-vector.patch)只修改上述两个 Ascend C 文件，完成两件事。

第一，`InitCausalMask()` 改为先用向量 `Duplicate` 将全部 `64×64` 元素置零，再逐行填充下三角的 1。
前 7 行的有效长度不足 8，使用 lane mask；其余行使用计数形式的 `Duplicate`。
初始化首尾保留 `PIPE_ALL` 屏障，清零与填 1 之间使用 `PIPE_V`，使掩码完整初始化并保留缓冲区切换屏障。

第二，64-token epilogue 在 `Exp` 后直接用 `Mul` 乘预置掩码，按 `gbrcStart * 64` 取得对应行。
原来逐行重建上三角零值的循环被移除，其中的短 Scalar `SetValue` 也随之删除。
这样，掩码的初始化和 epilogue 消费都走向量操作，避免在该位置混入短标量写。

失败候选也提供了边界清晰的对照：

- **仅加同步**：保留短标量写与向量操作交错的实现，对照仍失败，表明该同步候选不足以消除错误。
- **仅改初始化**：未移除 epilogue 中独立生成掩码的短标量写，因此仍无法修复输出。

初期候选保存在 a2；`failed-scalar-sync.patch` 只是失败记录，**不要应用到部署版本**。
最终修复没有修改 W4 数学、GDN 接受计数协议或精度门限。

## 5. 验证：同时检查输出、状态和精确掩码

修复前后，Prefill 输出相对 L2 误差如下；保留原记录的数值精度：

| Prefill T | Rank0 原始 | Rank0 修复 | Rank1 原始 | Rank1 修复 |
|---|---:|---:|---:|---:|
| 32 | 0.000448 | 0.000448 | 0.000450 | 0.000450 |
| 64 | 0.049745 | 0.000458 | 0.049599 | 0.000457 |
| 70 | 0.050862 | 0.000456 | 0.032159 | 0.000458 |
| 128 | 0.023383 | 0.000462 | 0.019197 | 0.000459 |
| 256 | 0.023539 | 0.000465 | 0.020019 | 0.000464 |

修复后两个 rank 共 **20 个 GDN 用例**通过原 `rtol=3e-3, atol=2e-3` 门限；Prefill 最大状态误差 `<0.000755`。
这些用例覆盖每个 rank 的 5 个 Decode 长度和 5 个 Prefill 长度，参考值来自独立 CPU 递推。
独立掩码在两个 rank 各重复 3 次，合计 **6 次均为 0 错误 / 最大绝对误差 0**。
修复后的 [GDN rank0](../../results/qwen38/tp2-serving-20261018/gdn-tp2-rank0.json)、[GDN rank1](../../results/qwen38/tp2-serving-20261018/gdn-tp2-rank1.json)以及[掩码 rank0](../../results/qwen38/tp2-serving-20261018/chunk-causal-rank0.json)、[掩码 rank1](../../results/qwen38/tp2-serving-20261018/chunk-causal-rank1.json)保留了原始指标。

[512-token GDN 补充门禁](../../results/qwen38/prefill-batch512-20261018/README.md)继续使用两个 rank、HK8/HV24、128 维、FP32 状态，并对照独立 CPU 递推：
输出相对 L2 为 `0.0004644–0.0004648`，最终状态最大绝对误差为 `0.0003618–0.0005109`，输出最大绝对误差低于 `0.000078`。
该轮保留的 6 次独立因果 mask 检查也全部零误差；掩码探针自身仍是 64-token 精确测试。

另对单一 `chunk_fwd_o` kernel 做首轮计时：`HK8/HV24/D128`，10 次 warmup、50 次设备事件计时，保留全部样本。

| T | 原始中位延迟 ms | 修复中位延迟 ms | 比值 原始/修复 |
|---|---:|---:|---:|
| 64 | 0.1271 | 0.1227 | 1.036 |
| 128 | 0.1903 | 0.1827 | 1.042 |
| 256 | 0.3750 | 0.3590 | 1.044 |

样本见[原始计时](../../results/qwen38/tp2-serving-20261018/chunk-kernel-bench-baseline.json)和[修复计时](../../results/qwen38/tp2-serving-20261018/chunk-kernel-bench-patched.json)。这组对照没有显示补丁引入明显性能回退。

## 6. 复现入口与已知限制

复现基线固定为 [opensensor/vllm@3ab5dda29acabea01f6a63d0806bdbbb4a27bde5](https://github.com/opensensor/vllm/tree/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5)与 [opensensor/vllm-ascend@ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032](https://github.com/opensensor/vllm-ascend/tree/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032)。
Catlass 固定为 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`；原环境使用 CANN 9.1.0、torch 2.10.0+cpu、torch-npu 2.10.0.post4。
应用补丁前保留原始 OPP，再在 Ascend fork 根目录应用仓库的 `gdn-causal-mask-vector.patch`，重编译算子及匹配桥接库。
原报告命令中的 `gdn-causal-mask-epilogue.patch` 与仓库存档名不同；这里以实际存档名为准。

在已配置算子环境的运行时中，复制两个探针到同一目录并执行：

```bash
export QWEN38_PROBE_OUTPUT=/experiment/gdn-patched
mkdir -p "$QWEN38_PROBE_OUTPUT"
torchrun --standalone --nproc-per-node=2 /experiment/qwen38_gdn_tp2_probe.py
# 512-token GDN 补充门禁，同时仍执行 64-token 精确掩码探针：
torchrun --standalone --nproc-per-node=2 /experiment/qwen38_gdn_tp2_probe.py --tokens 512
```

入口是 [qwen38_gdn_tp2_probe.py](../../tools/qwen38_gdn_tp2_probe.py)，它会调用 [qwen38_chunk_causal_probe.py](../../tools/qwen38_chunk_causal_probe.py)；后者通过 `run(rank, output)` 使用。
已有 a2 封装为 [run-gdn-tp2-on-a2.sh](../../results/qwen38/tp2-serving-20261018/run-gdn-tp2-on-a2.sh)：可用 `QWEN38_OPP` 切换原始/修复 OPP，用 `QWEN38_PROBE_OUTPUT` 分开结果目录。
该脚本会停止原 Qwen3.6 服务；[run-b512-gdn-on-a2.sh](../../results/qwen38/prefill-batch512-20261018/run-b512-gdn-on-a2.sh)也有此行为，并额外运行其他算子门禁。

已知限制：上述结果覆盖固定版本、设备和已测几何；流水线解释尚无底层指令跟踪证据。
计时仅覆盖合成输入下的单输出 kernel，排除了 WY、专家、PLE、通信等成本；原始实现存在正确性错误，其时间只用于观察补丁是否引入明显回退，不能推导整模 Prefill 提升比例。
该补丁目前只在本仓库 `patches/qwen38/` 中，没有记录过向 opensensor fork 或其他上游提交。
本问题与 Qwen3.6 的 GDN 状态行 bug（PR #18730）是不同缺陷，后者见 [Qwen3.6 GDN 修复记录](../qwen36/gdn-bug-fix.md)。
