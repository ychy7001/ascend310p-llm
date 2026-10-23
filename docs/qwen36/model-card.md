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
| INT8 KV / C8 | TP1/eager/MTP=0/4K短生成与chunked prefill已通过；32K配置已启动并完成约9.9K输入实测 |
| W4A8+MTP+C8 | TP1/eager与Graph、MTP1/MTP2在4K/32K/112K已通过启动、sanity、needle和阶段基准；TP2、全面质量与正式P95仍未验收 |
| 长上下文 | 无MTP TP1实际128K生成已测；256K配置不等于实际请求验收 |
| 多模态、全面精度、长上下文质量 | 未完成验收 |

独立公开checkpoint的启动/生成门禁见[release-validation](https://github.com/ychy7001/ascend310p-qwen36/blob/main/docs/release-validation.md)。数值算子门禁、基本生成与MTP接受率不能替代模型准确率评估。INT4可能因解包、路由、Prefill和通信成本更慢，未承诺10%–20%端到端收益。

最新实验快照（代码 `a486949`）：融合动态 KV 写入与按存储类型拆分的共享池已在真实 310P3 启动。W4A8+MTP+C8 的固定 GSM32 子集为 32/32；匹配 FP16 KV 对照为 31/32，其中唯一差异是 FP16 输出在 512 token 上限截断，不能解释为量化精度提升。TP1/eager/4K/MTP=1 的 C8 HTTP 吞吐（p512/o64、p3072/o32）中位数为 14.8362、2.9198 tok/s；对应 FP16 KV 为 15.5266、2.9430 tok/s。该结果包含 Prefill/HTTP，尚非纯 Decode 或正式尾延迟结论。

## 代码、构建与启动

公开代码：[ychy7001/ascend310p-qwen36](https://github.com/ychy7001/ascend310p-qwen36)。当前研究快照为 `529fc91`；[构建与启动说明](https://github.com/ychy7001/ascend310p-qwen36/blob/529fc91/README.md)、[公开制品验证](https://github.com/ychy7001/ascend310p-qwen36/blob/529fc91/docs/release-validation.md)和[C8实验说明](https://github.com/ychy7001/ascend310p-qwen36/blob/529fc91/docs/c8.md)可按固定版本查阅。

已测环境为Ascend310P3、CANN9.1、`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`、Torch2.10/torch-npu2.10.0.post4；宿主需要匹配的Ascend驱动与rootless Podman。复现依赖自定义OPP和Torch extension，单独下载权重无法直接用官方原版启动。

1. 获取代码和权重，在仓库根目录校验：

```bash
git clone https://github.com/ychy7001/ascend310p-qwen36.git
cd ascend310p-qwen36
git checkout 3688ebb
ms-hub download ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P --local-dir ./weights
python3 tools/validate_checkpoint.py ./weights --verify-hashes
```

2. 按仓库README准备上述310P镜像，禁用不兼容Triton，挂载仓库到`/project`和新的可写构建目录到`/work`。在容器内执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORK_DIR=/work/qwen36-build bash /project/scripts/build-native.sh
```

已验证对应源码的新目录分步编译、安装及真实NPU生成；尚未把公开build-native脚本在全新下载环境中逐字端到端验收。制品位于构建目录的`artifacts/opp/vendors/custom_transformer`和`artifacts/qwen36_int4_custom.so`。SDK和driver不随仓库提供。

3. 回到宿主仓库根目录，替换以下三个路径为宿主上的绝对路径，启动TP1、MTP=1、FP16 KV的4K首次复现配置：

```bash
MODEL_DIR=/absolute/path/ascend310p-qwen36/weights \
OPP_DIR=/absolute/path/qwen36-build/artifacts/opp/vendors/custom_transformer \
EXTENSION=/absolute/path/qwen36-build/artifacts/qwen36_int4_custom.so \
TP=1 MTP=1 MAX_MODEL_LEN=4096 MEMORY_UTILIZATION=0.90 PORT=18015 \
bash scripts/run-service.sh
```

`MTP=0`关闭推测解码；`TP=2`使用两颗芯片。默认eager；首次复现使用4K，长上下文需另做内存和质量门禁。启动器挂载只读自包含checkpoint，不依赖源BF16父目录。

4. 服务就绪后做生成检查：

```bash
curl http://127.0.0.1:18015/health
python3 tools/bench_generation_sanity.py \
  --base-url http://127.0.0.1:18015 --model qwen36-w4a8 \
  --out /tmp/qwen36-generation.json --max-tokens 128
```

sanity检查不等于准确率评估。当前C8为单独的`experimental/run-c8-service.sh`后端，需要额外构建C8库；TP1/W4A8+C8+Graph已在MTP1/MTP2下完成4K、32K和112K实机门禁，包含近上限needle；这些不是完整准确率集合或P95。TP2、正式上游Graph backend、多模态和32G实卡仍未验收。不要给标准启动器直接附加官方C8参数。动态scale在运行时生成，checkpoint不含静态KV标定scale。

## 发布校验与性能边界

2026-10-12完整上传64个文件、失败0；远端64文件大小和SHA256全部匹配本地，新下载config/index/manifest三文件逐字节匹配。没有重新下载全部权重再做推理。

匹配TP1/MTP=0/4K/.90、max-batch256的初版C8对照：p512/o64吞吐14.2881→13.2868tok/s，p3072/o32为2.9583→2.8152tok/s，均为1warmup/3rounds中位数，含Prefill/HTTP。当前C8转换路线更慢；有限生成检查不证明质量等价，容量收益仍需分配与长上下文实测。完整数据和边界见公开报告。

## 来源与许可

Qwen3.6源模型以Apache-2.0提供；见附带LICENSE及[官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md)。配套代码基于OpenSensor/matteius及vLLM贡献者的310P W4方案并增加Qwen3.6/ModelSlim/MTP适配，保留来源与版权声明。该制品不是华为、Qwen或vLLM官方认证版本。
