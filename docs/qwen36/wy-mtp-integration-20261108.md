# WY / MTP 组合验证（2026-11-08）

本轮仅在 a2 验证组合，不切换 a1，不推送，不改冻结 kernel、协议、规则、worker 或 R。镜像已有资产与本轮验证覆盖必须分开描述。

## 既有 TP1 镜像核查

已存在镜像：

```text
registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:8e17ec9ae646b1e418cf119bbb347238c4505b8b0037782a1345914824ec8d35
```

- 镜像 ID：`86e64c35bf5b13c732df580538684e98764c1257b68ca346398953a9b2ab4f68`。
- `/opt/qwen36/start.sh` 设置 `QWEN36_GDN_ENABLED=1`、`QWEN36_GDN_LIBRARY=/custom/pr17640_gdn.so`、`QWEN36_GDN_SOURCE=/gdn-source.py`，并传入 MTP speculative-config。
- `/c8bootstrap/sitecustomize.py` 加载 `pr17640_gdn_adapter`。适配器将固定 PR wrapper 的 `_C_ascend` 替换为独立 `pr17640_gdn` namespace，映射 WY 为 `compute`，并同时替换 chunk 模块、消费者和 package 的导出。
- `/gdn-source.py` 的 chunk 前向调用 `_compute_kernel_inputs_from_npu_wy`；不满足能力条件时仍可回退 Torch。因此启动开关本身不能替代实际 native eligibility 证据。
- wrapper SHA256：`3b9bc0eb2e2e9d944316858df1efaea72776e7221b5eb1640db4a719b985accb`。
- 该 wrapper SHA 与本轮 TP2 锁定镜像构建清单一致；binding 命名空间与二进制不同，不能据此宣称整套运行时等价。
- WY 主源文件 `chunk_gated_delta_rule_compute_wy.cpp` SHA256 为 `3f42fad80232e02a77e3995cd64dbb7af010275698dc3123db629336cbfac0c4`，亦与 TP2 构建清单一致；这不构成全部 include 文件或 OPP 二进制一致的证明。
- binding SHA256：`7a8004d28f5f488f36a142b3ec453ea1df5ac8396e7244d787ffa10bfbb92047`。
- a2 已有环境配置为 TP1、W4A8、MTP2、long-prefill、C8 关闭；不能将 a2 的配置直接当作 a1 当前部署状态。

历史验证并非空白：`mindieq-gdn-experiments-20261013.md` 记录固定 PR GDN + W4 + MTP2 的 8K/32K 重复、GSM32 为 31/32、114560-token 检索，以及 MTP1/2 对照。4096 调度批量下 8K Prefill 为 503.099 tok/s、Decode 为 37.100 tok/s；这是历史配置记录，不能当作本轮同条件 A/B，也不能把完整 GDN 收益单独归给 WY。

此前“W4A8 + WY + MTP 未测试”的笼统表述应修正为：历史路径与镜像已存在；缺少当前交付资产对应关系、实际 native 路径确认及本轮组合对照。

## 本轮 TP2 W8A8 首次结果

使用锁定候选 `localhost/qwen36-pr17640-wy-locked:20261108`，image ID `9df3c6f3e42084d77ef258853cf824a011b4d30a1b05acc72a834a691c073163`。

在原 TP2 manifest 上仅增加 MTP1，保留 FULL_DECODE_ONLY、capture `[1,2]`、prefix cache 和其余设置。主模型和草稿模型加载成功，但模型图捕获失败。调用栈指向：

```text
_310p/attention/attention_v1.py -> forward_chunked_prefill_310
_310p/attention/attention_mask.py:95 -> query_start_loc.to("cpu", dtype=torch.int32)
aclrtMemcpy 107030 / captured-stream synchronization 107027
```

这是图捕获兼容性失败；没有请求质量或组合性能结果，不能归因为 WY 精度失败。实验容器已停止、日志保存、删除，独立快照确认 NPU 无运行进程。

结果目录（a2）：`/home/ychy/ascend-lab-runtime/hyperqwen-20261103/wy-mtp-matrix-20261108-v1/`。总耗时 231.316 秒。下一步仅关闭图执行做 MTP1 smoke 隔离；即使 eager 通过，也不能证明原 Graph 组合可交付。

## TP2 eager 最小隔离结果

同一镜像、同一模型和原 prefix 设置，仅移除 compilation-config 并增加 `--enforce-eager`。MTP1 启动成功，三条生成分别返回 `OK.`、`42` 和包含 `57.80` / `2026-11-08` 的合法 JSON，均正常 stop。耗时 217.445 秒；独立快照清场通过。

结果目录（a2）：`/home/ychy/ascend-lab-runtime/hyperqwen-20261103/wy-mtp-eager-smoke-20261108-v1/`；最小结果已归档至仓库 `results/wy-mtp-integration-20261108/tp2-eager-mtp1/`。这是基础可运行证据，没有完成 PDF、长输出或正式质量验收，不支持 eager 生产放行或性能收益声明。原 Graph 启动失败结论不变。

## 后续矩阵

1. TP2：eager MTP1 最小生成隔离已通过。先定位 SplitFuse mask 在 Graph capture 中的主机拷贝问题，再决定 Graph MTP1/2 性能/PDF；不为绕过 Graph 失败而将 eager 设为交付默认。
2. TP1：使用已有镜像及其原始 wrapper 的 Torch/native 两函数，固定其余 GDN 与 W4 组件；比较 WY Torch/native × MTP0/2。观察适配器打印实际 native eligibility；不替换 kernel。
3. 性能只做同一量化格式和同一执行模式的比较。记录冷 Prefill、TTFT、Decode、MTP counters、PDF 输出和清场；未经质量验证不升级默认交付配置。
