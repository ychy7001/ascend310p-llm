# M=64 Prefill kernel candidate（2026-10-13）

这是一次隔离的 Ascend C kernel 实验，未进入默认源码和稳定产物。候选在 `qwen_w4_a8_int4_matmul_v310.cpp` 中针对 `4096 <= numRows <= 16384` 选择 `M=64` schedule；重新编译 310P OPP、合并稳定 pack/SwiGLU OPP、加载独立 extension，并完成真实服务启动。

结果：

- 3072/1、4K maxlen（实际未命中 M=64，旧分支）：Prefill 中位约 `379.87 tok/s`，needle 通过。
- 4096/1、8K maxlen（实际命中 M=64）：`376.76/376.60/376.80 tok/s`，中位 `376.76 tok/s`，计数有效。

4096 结果与默认 1024/512 的历史 3072/1 基线不是完全同条件（max_model_len 不同），且没有超过既有水平，因此不保留该分支。主要成本是更大的 UB/activation tile，未显示足够收益。a2 独立产物路径：`qwen36-public-release-20261012/m64-artifacts2`、`m64-ext`、`m64-combined3`；服务已经停止。
