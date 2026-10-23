# 来源、修改与精度契约

社区参考：[opensensor/vllm-ascend](https://github.com/opensensor/vllm-ascend)，基于`0637d3baa6bc2000ba6433b301f233c6516847b4`。感谢matteius及vLLM贡献者的310P W4实现。原文件SPDX与版权声明保留，没有将原作者填写为新增提交的Co-author。

`runtime/vllm_ascend/models/qwen4_exp/`为经过a2测试的社区模块子集，并非整个社区模型。`w4_moe.py`、`w4_backend_policy.py`等含本地ModelSlim metadata/backend适配；非专家执行和FP16驻留策略亦有实验修改。`native-overlay/`为a2实际构建源码的三个算子家族快照，不是未经修改的社区commit：包含per-token activation pack、融合SwiGLU pack和per-channel跨K累加。来源revision只能标识基础。`runtime-source-sha256.json`固定首发历史快照，不会随着后续调优重写；当前N128/C8/MTP/Graph核心源码与启动器清单见`source-snapshot-20261013.json`，结合Git历史追踪修改。

`runtime/vllm_ascend/models/qwen36_w4_patch.py`为独立Qwen3.6适配：替换routed experts、处理checkpoint加载、TP专家所有权和BF16 MTP。公开首版采用已验证的普通W4路径，不启用未通过广泛性能门禁的fused down-reduce候选。`tools/`中的量化与测试、`experimental/`中的C8为独立实验模块。

源权重来自ModelScope `Qwen/Qwen3.6-35B-A3B` BF16 checkpoint。原始下载的immutable revision尚未完整记录，不能以今天的远端main代替量化源revision；本发布checkpoint使用逐文件SHA256固定制品，并明确该来源追溯限制。未来重新量化应先锁定源revision和源文件哈希。

首版权重量化为ModelSlim SSZ对称per-output-channel INT4；**不是每个128元素组独立拟合scale**。每通道scale按G128 kernel ABI重复存储，offset为FP16零，权重按输入轴低nibble优先打包。激活为设备端per-token INT8，再以INT4 limbs组合执行Cube整数计算。不能将其写成直接原生INT8×INT4单指令。

仅40层routed experts使用W4；共享专家、attention、embedding、LM head及vision等原始浮点张量保留，由runtime按其策略消费。Qwen3.6自身19个MTP tensor保留BF16，包含两个stacked expert张量，非借用Qwen3.5 MTP。text-only生成已测；图像输入、完整质量和全部长上下文场景未验收。

模型官方[README](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md)声明Apache-2.0。本仓库以Apache-2.0提供代码；CANN构建依赖从固定社区checkout及用户已安装SDK获得，不打包SDK或driver。

当前 orangepi 的实验镜像及其来源边界、以及后续统一镜像的分层构建方案，见
[`docs/qwen36/runtime-lineage.md`](qwen36/runtime-lineage.md)。当前镜像仍包含不可从 Docker
metadata 完整恢复的本地构建内容，不能把 `local/vllm-ascend` tag 单独当作可复现
版本；正式发布前必须使用源码 commit、镜像 digest、CANN/driver/SOC 和算子制品
哈希共同标识运行时。
