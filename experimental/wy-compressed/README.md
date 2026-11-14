# WY 与 compressed Attention 组合验证

WY 不依赖 legacy eager。WY 加速 GDN/linear attention；compressed SplitFuse 用于 full attention。本目录验证二者组合，并保留升级时发现的独立算子缺陷。本轮结论见 [验证记录](../../docs/research/wy-compressed-attention-20261115.md)。当前正式默认仍为已通过验收的 20261114 WY 版本。

## 无模型最小复现

`repro_future_mask.py` 只依赖 PyTorch/torch-npu，不导入 vLLM、不加载模型、不调用 WY。固定零 Q/K，将指定 query 因果边界之后的 14 个 V 从 0 改为 64。该行输出必须仍为 0；最后一行必须变化，用于排除 KV 写入没有生效的伪通过。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 repro_future_mask.py --device 0
```

- 物理 KV block 为 64，head_dim=256。模型的逻辑 block=640 会拆为十个物理 block；直接给算子传 640 是无效探针。
- context=32768 是通过对照；32774 是失败例。脚本包含 legacy 对照和 compressed 被测组。
- 检出未来值影响时退出 1，表示成功复现缺陷。旧运行栈缺少 v2 时仅检查 legacy，并明确输出 `compressed_available=false`。
- 不以放宽误差阈值或文本看似正常作为通过依据。

## a2 复核入口

`run_numeric.sh` 获取 `/tmp/a310-npu.lock`，要求 a2 无活动容器/NPU 作业，依次检查两张卡，保存前后 NPU 状态及退出码。整个工作目录挂载到 `/bench`。

```bash
NUMERIC_PREFIX=future-upstream-64 \
NUMERIC_PROBE=repro_future_mask.py \
NUMERIC_IMAGE=b217eec3ca1772d3836f0db3772255866fcd5f5397b0b5bb5451f04fc9b0d185 \
./run_numeric.sh
```

指定新的 `NUMERIC_PREFIX` 保存新一轮证据；执行器会拒绝覆盖已有日志。`NUMERIC_IMAGE` 为本机预存镜像 ID，需要先确认相应软件栈。`attention_numeric.py` 是较大的真实算子随机输入对照，提取候选镜像实际函数体和冻结的 `legacy_attention_mask.py`，覆盖单请求、混合/重排、每序列 3 个 query 的 MTP2 验证形状与 context 边界；用完整输出对照和最差行 CPU FP32 参考检验，并诊断与错误因果边界的接近程度。其历史 640 探针为无效输入，不能作为精度结论。

## 双路径服务筛查

`prepare.py SOURCE OUT` 生成两个一次性路由日志 overlay；输入为固定候选镜像中的 `_310p/` 源码。日志在真正的 native WY / compressed SplitFuse 调用返回后打印，不修改计算结果。`launch.sh` 保持 WY + MTP2 + Decode Graph；`screen.sh` 检查实际双 rank 命中、Graph 回放、MTP drafts、32K 检索以及正常清场。

```bash
GRAPH_SIZES=12 ./screen.sh graph12-mtp-262k
```

这只能证明服务组合可运行。262144 是配置上限，32K 筛查没有证明完整 262K 功能或性能。三档 Graph `[3,6,12]` 在该新栈上事件资源耗尽；一档 `[12]` 通过。新栈的 compressed 算子因果门禁失败，未进入性能放行或镜像发布。
