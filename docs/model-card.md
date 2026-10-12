---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
pipeline_tag: text-generation
tags:
  - ascend
  - ascend310p
  - w4a8
  - mtp
  - experimental
---

# Qwen3.6-35B-A3B W4A8 SSZ + BF16 MTP for Ascend 310P

这是Qwen3.6-35B-A3B的实验性自包含checkpoint：routed experts采用signed INT4对称per-output-channel SSZ，activation由配套算子动态量化为per-token INT8；其余模型与MTP张量保留原始浮点数据。需要配套[Ascend310P runtime](https://github.com/ychy7001/ascend310p-qwen36)，不能假设原版Transformers或官方vLLM直接加载本量化ABI。

## 制品与精度契约

- 源权重：ModelScope [Qwen/Qwen3.6-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B)，BF16；不从社区W8A8重新量化，也不借用Qwen3.5 MTP。
- 40层×256 experts×3 projections，共30,720个expert projection；packed weight、scale、offset共92,160个张量。
- scale为FP16 per-channel，并按G128 kernel ABI重复存储，**不是每个G128独立量化**；offset为FP16零。低nibble对应先出现的输入轴元素。
- 激活per-token INT8用两组signed INT4 limbs组合Cube整数计算，不声称直接INT8×INT4单指令。
- Qwen3.6自身19个MTP张量为BF16，包含stacked gate/up和down expert张量。
- 52个safetensors分片，总93,125个indexed tensors。checkpoint文件约24.63GB十进制，tensor payload为24,591,896,288 bytes；这不等于服务运行时NPU显存。
- 原始源checkpoint immutable revision追溯仍不完整；当前制品用`deployment-sha256.json`固定。不要把当前远端main当成既有量化来源revision。

## 已测与未测

| 项目 | 状态 |
| --- | --- |
| 完整W4A8 TP1/TP2、FP16 KV | 已在真实310P3运行 |
| BF16 MTP | TP1/TP2、num_speculative_tokens=1已运行 |
| INT8 KV / C8 | 独立算子实验，尚未完整模型接入 |
| W4A8+MTP+C8 | 尚未联合验收 |
| 长上下文 | 无MTP TP1实际128K生成已测；256K配置不等于实际请求验收 |
| 多模态、全面精度、长上下文质量 | 未完成验收 |

独立公开checkpoint的启动/生成门禁见[release-validation](https://github.com/ychy7001/ascend310p-qwen36/blob/main/docs/release-validation.md)。数值算子门禁、基本生成与MTP接受率不能替代模型准确率评估。INT4可能因解包、路由、Prefill和通信成本更慢，未承诺10%–20%端到端收益。

## 使用

```bash
ms-hub download ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P --local-dir ./weights
python3 tools/validate_checkpoint.py ./weights --verify-hashes
```

配套仓库中构建独立OPP与Torch extension，然后运行`scripts/run-service.sh`。已测环境为Ascend310P3、CANN9.1、`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`、Torch2.10/torch-npu2.10.0.post4。初次建议TP1/4K/eager，再按容量与质量门禁扩展。

本方案C8为运行时动态量化，不包含静态KV标定scale；因此本checkpoint不是官方静态C8 checkpoint。

## 来源与许可

Qwen3.6源模型以Apache-2.0提供；见附带LICENSE及[官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md)。配套代码基于OpenSensor/matteius及vLLM贡献者的310P W4方案并增加Qwen3.6/ModelSlim/MTP适配，保留来源与版权声明。该制品不是华为、Qwen或vLLM官方认证版本。
