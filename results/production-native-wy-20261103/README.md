# 原生 WY 单变量实验（2026-11-03，持续更新）

目标是在 a2 的生产同版本镜像上，仅以原生 ComputeWy 替换 Torch WY helper。Graph[1,2]、TP2 W8A8、MTP0、262144 上下文、Prefix Cache 和生产 FwdH/FwdO 保持原配置。a1 未修改。

复用 GruntovDima 的 Ascend PR17640，固定 commit `18163f6b1d208ab3b4fb06e9752225611c37f40e`；源码 4536 个 Git blob 校验通过，archive SHA256 为 `2fe2a12e45afa62506d371858081176c62a21b78f13456afed4127702072c3fc`。本地源码位于 workspace/reference/ascend-pr17640-18163。OPP 与独立 namespace 绑定由生产镜像构建，包的 API/文件检查不含 FwdH/FwdO，原始记录见 package-isolation.json。

构建与诊断不是模型端到端性能结果。模型实验只有在 numerical-gate.jsonl 的 10 项门禁通过后启动。门禁包括独立 CPU 逐 token 递推参考、尾块、变长、非零初始状态、不同头数与维度、强衰减和候选逐位重复检查。

## 发现的原路径数值偏差

在 synthetic 输入（seed311，111 tokens，Q/K heads8、V heads16、head dim128、FP16 非零 state、负 gate 范围[−1,0]）中，生产原 Torch WY 路径出现偶发输出偏差。独立 CPU 递推参考已先与生产 CPU chunk 参考核对，17/71-token 两组误差约 2e−6。

order-diagnostic.jsonl：同一输入 12 轮交替执行，原路径 7/12 在固定门限内，WY 候选 12/12 在门限内；候选最大绝对输出误差均约6.32e−6，原路径最大约2.14e−3。单次六形状诊断中，该样本原路径 relative RMSE约3.17%，候选约0.043%；其余五组两路径均接近参考。诊断指标不替代正式门禁，也不证明已定位具体内核根因或业务影响。

初始严格门禁要求基线也通过独立参考，因此在这个原路径问题处停止。修正后的策略保留基线失败详情，候选始终满足相同独立参考门限（逐元素 atol=2e−4、rtol=.04，整体 relative RMSE≤.02）和逐位重复检查；仅当基线也与独立参考吻合时才强制候选/基线一致。没有放宽候选质量标准以追随错误基线。WY layout 的 W/U/G 子项另有显式容差，详见脚本和每条记录。

## 失败记录与复现

- preparation-failed：CPack 的同名 `.run` 目录被误列作包，修复为只选择普通文件。
- registration-missing / bootstrap-missing：独立算子测试没有按生产流程 bootstrap 原 custom ops，分别出现未注册和原 FwdH API 查找失败；补齐生产 manifest 环境并在设备初始化前调用 enable_custom_op。
- baseline-reference-failed：原路径对独立参考的实际数值失败；保留首个通过样本与失败堆栈，另用 diagnostic/order-diagnostic 定位范围。

这些准备与诊断失败均未启动模型 Benchmark。各次日志、命令、状态和部分数值输出保留，未改写成成功结果。

执行入口：benchmarks/ascend310p/build-native-wy.py、validate-native-wy.py、run-native-wy-screen.py。远端工作目录 /path/to/lab-runtime/hyperqwen-20261103/native-gdn，模型候选目录 production-native-wy、端口18108。原始 tensor snapshots 仅保留 a2，供后续最小复现；本目录保留较小的指标和日志。

## 模型初轮结果与验收状态

初轮 B 已完成。四档 Prefill +38.42%/+54.02%/+46.06%/+39.54%，Decode −0.93%/−0.80%/−0.49%/−0.72%；C1/3/4 stream +2.63%/+8.36%/+6.63%；26请求/20690输出 tokens 长回放492.81s（原508.40s）。GSM64 63/64、1条截断/缺最终标记，与原汇总相同。原始请求、完整输出及自动状态均保存，不从 status.json 的 promising 推导业务验收。

**人工辅助语义审阅阻止接受候选**：business-semantic-review.json 记录会议4负责人严重错配；原基线正确。28项自动门禁未覆盖此类语义错误。正在运行独占数值诊断与 B/A′ 会议负责人配对复测，并重测原配置性能；结果未完成前不推荐部署或称已修复。

cpu-oracle-111.json：同一份111-token失败输入上，生产 CPU chunk 与独立逐 token 参考输出最大绝对误差3.59025e−6、状态5.21541e−8。它只核对 CPU 参考，不能替代空闲 NPU 复核。

## 配对证据修正

原配置 A′ 在三并发整套业务中也出现相同负责人轮换错配；两配置串行各6次均正确。首个失败仍保留，但不能再称“候选独有回归”。最新判定见 semantic-adjudication.json，当前不升级部署，等待原三并发重复和数值拆分结果。空闲数值复测原路径3/12、候选12/12，最终状态均逐位稳定、偏差只在token56；诊断在 results/production-wy-control-recheck-20261103/。

## 数值拆分与Profile更新

A′长回放508.42s，A508.40/B492.81，Prefill收益与约3.1%业务墙钟收益经A/B/A′对照仍成立。三并发复测3轮与双rank Profile完整归档至results/production-wy-batching-recheck-20261103；原生ComputeWy确实执行。

**数值判定修正**：候选在最初12轮稳定仅是当时的调用序列。后续冻结其WY输出并重放下游，仍出现3/12超限。WY与FwdH边界逐位稳定时，FwdO原始输出也会偏离；不能将原生WY记作数值修复。固定地址FwdO-only24/24通过与完整链路偏差并存，完整链路/原始输出复核支持继续检查FwdO。mask补丁仍独立构建验证，原OPP未覆盖，候选不升级部署。
