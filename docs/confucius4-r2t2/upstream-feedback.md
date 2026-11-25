# 上游反馈准备

状态：Issue #13 已提交，PR 等维护者确认协议语义后再决定。

## 建议优先提交

### Confucius4-R2T2 Issue：WebSocket v1 VAD reset 丢失同帧增量

- 草稿：[upstream-vad-reset-issue.md](../../../ascend-lab/reports/confucius4-r2t2-310p/upstream-vad-reset-issue.md)
- 影响：开启 VAD 时，VAD speech-end 与同一 chunk 的稳定后缀同时发生，客户端可能丢掉一个或多个已经生成但尚未发送的字符。
- 复现：公开 `resources/test.wav` 加 1 秒静音、160 ms PCM 实时发送、CUDA、`/asr_stream_api_v1`、`use_vad=true`。
- 证据：关闭 VAD 保留“酒水”；开启 VAD 重复出现“带酒。”；服务端已生成“带酒水。”。补丁使 v0/v1 都发送“水。”，关闭 VAD 的 v1 对照不变。
- 本地材料：[v0/v1 最小补丁](../../patches/confucius4-r2t2/preserve-vad-reset-delta.patch)、[无依赖回归](../../../ascend-lab/experiments/confucius4-r2t2-310p/check_vad_reset.py)。原版失败 4/10，补丁版通过 10/10。
- 方案待维护者确认：A 为已验证的单帧补丁，合并同帧增量；B 为未验证的双帧方案，普通帧发增量，reset 帧保留原有尾缀。A 改变 reset 载荷，B 改变帧数。
- 已提交：[Issue #13](https://github.com/netease-youdao/Confucius4-R2T2/issues/13)；PR 等维护者确认协议预期后再决定。

### 为什么值得向上游反馈

- 缺陷在发布方自己的 `ws_server.py` 消息组装，不在 310P、CANN 或任何算子层。CUDA 实机 2/2 复现，纯 CPU 的源码级回归也能独立复现，与我们的设备栈无关，属于应回上游的通用缺陷。
- 影响面覆盖所有可能遇到该时序的 VAD 输入：一旦某个 chunk 同时推进稳定文本并触发 VAD 结束，客户端累计文本就可能永久缺字，而服务端日志里文本是完整的。用户很难自行判断原因。
- 复现成本低：公开样例加 1 秒静音、公开权重和 VAD checkpoint、三十行无依赖脚本即可验证，维护者不需要 Ascend 环境。
- 修复面极小（两个 reset 分支），但 `reset` 载荷语义需要发布方定义，先 Issue 后 PR 是最低摩擦路径。
- 上游仍活跃且接受反馈：仓库 902 stars，2026-11-16 仍有推送，维护者回复过 #8；现有 Issue 中无 VAD/reset 同类报告，不构成重复。
- 我们代交付要长期维护这个补丁。上游修掉可以去掉下游分叉，避免每次升级都重放。

## 暂不向 vLLM-Ascend 提交

- `streaming_transcribe()` 的 `fixed_text` 候选回退在 CUDA 与 310P 均出现，属于 R2T2 SDK/模型侧语义，不是 Ascend 算子问题。
- `language` 与 `to_language` 的 prompt 兼容问题来自当前 Qwen-ASR/vLLM 入口组合，先由下游维护。
- 310P 专用运行时、W4A8、WY 和 CANN workaround 保留在 `ascend310p-llm`。

## 提交前检查

- [x] 负责人确认 Issue 标题、复现步骤和修复方向。
- [x] 2026-11-23 核对公开 Issue，未发现 VAD/reset 同类报告；#8 是尾部 `|` 问题。
- [x] 默认先提交 Issue，未附 patch；PR 等维护者反馈。
- [x] 未附权重、音频文件、私有日志、内部路径或设备信息。
- [x] 提交前再次读取远端 issue 列表，确认无 VAD/reset 同类条目。
