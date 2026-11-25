# Confucius4-R2T2 310P

状态（2026-11-23）：原样权重可在单逻辑 310P3 加载。专用转写接口已通过七输入 smoke，是离线候选；聊天接口的自动语言输出未通过。同一流式样本在 310P 和 CUDA 均出现 SDK 候选回改，流式未作为交付档发布。
单独维护语音版本线，不替换本仓库默认 Qwen3.6 镜像或服务。

## 版本与制品

- ModelScope：`netease-youdao/Confucius4-R2T2`，权重 revision `7eb29d6aa1779829512babab1825844b93ff43e2`。
- [推理文件锁](../../configs/confucius4-r2t2-modelscope.lock.json)记录 12 个文件的 revision、大小和 SHA256，排除训练状态。模型许可与示例代码许可分别以发布方为准，不分发权重。
- 运行线：`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，digest `sha256:16770d322d0e343bc5ff61f897acfc6fb1383eee56d7939554e1c9cf8b504773`。FP16、TP1、eager，不启用实验补丁。
- 上游支持边界：vLLM-Ascend 当前文档将 Qwen3-ASR 标为自 v0.30.0 停止维护。此语音适配不依赖上游恢复该维护项；在受支持版本上能独立复现的通用缺陷仍可单独贡献。

从本仓库根目录下载到一个不存在的制品目录：

```bash
python3 benchmarks/ascend310p/download-pinned-modelscope.py \
  --repo netease-youdao/Confucius4-R2T2 \
  --file-list configs/confucius4-r2t2-modelscope.lock.json \
  --out /path/to/confucius4-r2t2-7eb29d6a
```

加载子目录 `model/`。离线验证可复用 `ascend-lab/experiments/qwen3-asr-benchmark/scripts/serve_container.sh`，指定单设备和独立端口。现有 launcher 为了复用 benchmark 仍暴露 `qwen3-asr` API 别名；该别名不代表权重身份。

## 离线与流式入口

- 离线候选：`POST /v1/audio/transcriptions`，multipart `file`、`model` 和 `response_format=json`。七输入两次独立启动共 28/28 非空。优先验证该入口的重复性与质量，不以 `POST /v1/chat/completions` 替代。后者七输入诊断有 24/84 没有正文，另 24/84 的标签为 `language None` 但有正文。
- 并发重复性合计覆盖三次 FP16 eager 冷启动及并发 1/2/4。126/126 非空成功，122/126 逐字一致，四次差异来自同一个 AMI 片段的标点和大小写。严格逐字门禁待通过。
- 同一场 AMI 英文会议的前 60 秒和 300 秒各成功转写一次，诊断 WER 分别为 43.55% 和 21.34%。完整质量回归仍在实验仓推进。
- 固定镜像的 Qwen3-ASR 生成前缀读取 `to_language`，不读取转写请求的 `language` 来构造该前缀。不要把两者互换或据此承诺强制语种。
- 流式候选：保留官方 `fixed_text` 与 flush 原始证据；对外区分可替换的 `partial` 和权威的 `final`，不可直接按长度把异前缀拼到旧文本。严格 `commit` 需在真实节拍、多输入和 reset/EOS 下单独通过门禁。
- 独立缺陷：官方 WebSocket v0/v1 在同一 chunk 的 VAD reset 可能覆盖尚未发送的文本。[下游实验补丁](../../patches/confucius4-r2t2/preserve-vad-reset-delta.patch) 只修正消息组装；CUDA v0/v1 与 310P v1（CPU VAD）单流单样本通过。补丁未进入默认镜像或发布档，也不解决 SDK 候选回改。[上游反馈草稿](upstream-feedback.md) 待确认后提交。

## 维护边界

`ascend-lab` 保留实验探针、基准及证据；通过验收后，本仓库维护部署档、版本锁和必要补丁。R2T2 的稳定前缀属于模型侧，先验证官方状态机和 310P 性能，不新增 Ascend 模型类。其它语音模型逐个适配，不能从这次 smoke 推断兼容。不要使用 `pip install confucius4-r2t2` 自动解依赖；该路径可经 `qwen-asr[vllm]` 替换镜像的 vLLM。
