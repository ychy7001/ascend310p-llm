# vLLM Ascend 上游提交包

状态：可提交，尚未在上游发布。先开 [issue](issue.md)，再用 [PR 文案](pr.md) 提交 draft PR，在正文加入 issue 编号（`Related to #...`，不要用 `Fixes`）。

补丁以核对时的 vllm-ascend `main` `4b3bd7ebe51094f3afa5288bb7a3316d12d3154e` 为基线，已生成带 DCO sign-off 的本地提交 `278ad7293`。可在基线上用 `git am 0001-310p-splitfuse-causal-repro.patch` 导入。[脚本](310p_splitfuse_causal_repro.py)和[双卡原始数值](a2-operator-results.json)可供 issue 单独引用。诊断只使用合成输入，不依赖内部环境路径或权重。

PR 只加按需运行的复现；不改推理路由、不放宽门禁、不提交未经验证的生产 fallback。上游确认算子合同或修复后，再考虑将同一用例转为自动回归。
