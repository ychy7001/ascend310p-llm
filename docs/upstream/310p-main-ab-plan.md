# 310P legacy SplitFuse：上游 main 对照协议

- A：`vllm-ascend 76ea9d24f155afe032ad9ac6c98717f1f7334f36`，该提交 `Dockerfile.310p` 默认配对的 vLLM `v0.33.0`（解引用为 `4ba787ec9959e566d3b587a796dff07673324da5`），CANN 9.2.1 amd64 基底 `sha256:c4844ab1f074b526c222ea86b6cc55f0f41152c9fc20d6f49a62b259c26dadf2`；不引入验收镜像的 #17640、#18730 或 Graph overlay。
- B：A 的同一镜像，只覆盖补丁修改的 310P Python attention 和 mask 文件。先核对 Python 包加载路径、实际算子分支和 CLI，确认唯一变量是 active-width mask。若新版 CANN 选择压缩 SplitFuse，正常路由用作对照；另用两臂共同的诊断开关强制 legacy，并明确标为强制模式。不能把未经过目标路径的 A/B 当作优化实测。
- 两组先用可共同启动的 `max_model_len` 做 eager、TP2、batch 2048、prefix on、MTP off，分别测首段 FlashAttention、冷分块、prefix hit 和混合批次。先以 4K/8K 和较短上限核对合法原路径，再提高到两臂共同可运行的长档；如果初始 FlashAttention 的大 mask 先阻塞，记录错误和内存，不用旧镜像补丁把它悄悄绕开。
- 验证同一 token ID / 合成 fixture、输出与检索、前缀实际命中。性能窗口不用 profiler；单独采样并记录 legacy mask 的 shape、分配/峰值内存、TTFT 和覆盖分支。Graph/MTP 独立做回归，原版不能运行时如实记录，不把它计为 B 回归或修复。
- PR 主表只用同 main 镜像 A/B；旧 `bb0fa66f…`（native WY #17640、配套状态和 Graph overlay，本地集成）数据只能作为单独背景，不用于首个分配补丁的性能主张；跨层复用也另起补丁和主表。核对到的 #17640 仍开放（2026-11-14）。

`.github/vllm-main-verified.commit` 的 `24420b7a341af37a9bc3186e6fc6062512e0aa58` 是上游 main 跟踪点。它用在 lint 镜像流程；发布镜像 workflow 使用根 Dockerfile 的默认依赖。初次构建在 Rust 工具链安装时主动结束，保留 `build.log`；其临时安装的 CPU Torch 版本不能用来判断最终 NPU 镜像是否兼容。首次使用正式配对时，容器内 GitHub fetch 遇到 TLS 中断（`build-release-fetch-failed.log`）。直接 Git 重试的 pack 也不完整。改从 GitHub 下载固定 SHA 的源码归档，使用从 pack 取出的原 commit/tag 对象，校验源码 Git tree 和完整性，恢复同一 `v0.33.0` 浅库。通过 `VLLM_REPO` 只读挂载，保留原 Dockerfile 的默认标签。vLLM 源码恢复和 Rust 构建已通过。310P 编译首次失败因为 Ascend GitHub 归档未递归包含 Catlass；已从 a2 Git 缓存导出上游锁定提交 `41bf90da655bba3c66d0acd7e00abe33960ecfd6` 的源码，重试同一 Dockerfile。正式构建写 `build-release.log`，待校验 commit 和依赖后才写 `image-id.txt`。

同一台 a2 曾在旧版官方 v0.30.0 / CANN 9.2.1 组合中观察到 GDN 数值稳定性异常（[归档](../../results/cann921-upgrade-20261104/README.md)）。它不代表当前 `main`；若 Qwen 启动或输出门禁失败，先定位两臂共同的运行时问题，不能把差异归因于 mask，必要时用模型无关的 legacy 算子结果单独反馈。

构建工件仅在 a2 实验目录 `~/ascend-lab-runtime/upstream-main-76ea9d24`，如构建失败保留 `build.log` 等定位依据。发布到上游前冻结最终镜像 ID、基底 digest、运行时依赖和实际硬件算子路径。
