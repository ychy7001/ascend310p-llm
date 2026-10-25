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

Qwen3.6-35B-A3B 的自包含量化 checkpoint，面向 Ascend 310P3。Routed experts 量化为 signed INT4（ModelSlim SSZ，per-channel 对称），激活在算子内动态量化为 per-token INT8；其余张量（attention、norm、MTP）保留原始 BF16。

配套运行时是本仓库 [ychy7001/ascend310p-llm](https://github.com/ychy7001/ascend310p-llm)，官方 Transformers / vLLM 不认识这套量化格式，需要用仓库里的自定义算子加载。

## 量化格式

| 项目 | 数值 |
| --- | --- |
| 量化范围 | 40 层 × 256 experts × 3 projections = 30,720 个 expert projection |
| 打包后张量数 | weight + scale + offset，共 92,160 个 |
| scale | FP16，per-channel，按 G128 kernel ABI 重复存储（不是每个 G128 独立量化） |
| offset | FP16，固定为 0 |
| 激活量化 | per-token INT8，运行时用两组 signed INT4 limbs 组合做 Cube 整数计算（不是硬件原生 INT8×INT4 单指令） |
| MTP | Qwen3.6 自带的 19 个张量，BF16，不量化 |
| 文件 | 52 个 safetensors 分片，93,125 个 indexed tensors，约 24.63 GB（tensor payload 24,591,896,288 字节，不等于运行时显存占用） |

源权重来自 ModelScope [Qwen/Qwen3.6-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B)（BF16）。不是从社区 W8A8 版本重新量化，MTP 也不是借用 Qwen3.5 的。源 checkpoint 的 immutable revision 未完整记录，当前制品以仓库内 `deployment-sha256.json` 为准，不代表今天 ModelScope 远端 main 分支就是原始量化来源。

## 验证结果

单流、固定协议测量，不代表 P95 或完整精度评测；完整数据见[发布校验记录](https://github.com/ychy7001/ascend310p-llm/blob/ab1a370/docs/qwen36/release-validation.md)。

| 能力 | 结果 |
| --- | --- |
| W4A8 推理，TP1/TP2，FP16 KV | 真实 310P3 跑通 |
| BF16 MTP（1 枚草稿） | TP1/TP2 跑通 |
| INT8 KV（C8） | TP1/eager/MTP0，4K 短生成和 chunked prefill 通过；32K 配置跑通约 9.9K 输入 |
| W4A8 + MTP + C8（Graph） | TP1，MTP1/MTP2，4K/32K/112K 全部通过启动、sanity、长上下文 needle 检索 |
| 长上下文（无 MTP） | TP1 跑通 128K 实际生成 |
| GSM8K 固定 32 题子集，W4A8+MTP+C8 | 32/32 |
| 同配置 FP16 KV 对照 | 31/32（唯一差异是输出在 512 token 上限截断，不是量化精度问题） |
| C8 吞吐（TP1/eager/4K/MTP1，中位数） | p512/o64 14.84 tok/s、p3072/o32 2.92 tok/s（对应 FP16 KV 为 15.53、2.94 tok/s，含 Prefill） |

未做：TP2 的完整质量评测和正式 P95，多模态，长上下文下的完整精度评测。数值算子门禁和 GSM 子集结果不能代替完整准确率评估；INT4 不保证比 FP16 更快，实测吞吐已如上列出，供自行判断。

## 快速开始

环境：Ascend 310P3、CANN 9.1、镜像 `quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`、Torch 2.10 + torch-npu 2.10.0.post4。需要自定义 OPP 和 Torch extension，单独下载权重无法直接用官方原版 vLLM 加载。

下面的启动命令使用 rootless Podman；用 Docker 的用户需要自行调整设备挂载和用户映射方式，仓库未在 Docker 上验证过启动流程。

**1. 下载代码和权重**

```bash
git clone https://github.com/ychy7001/ascend310p-llm.git
cd ascend310p-llm
git checkout ab1a370
ms-hub download ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P --local-dir ./weights
python3 tools/validate_checkpoint.py ./weights --verify-hashes
```

**2. 构建自定义算子**（容器内，挂载仓库到 `/project`，可写构建目录到 `/work`）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORK_DIR=/work/qwen36-build bash /project/scripts/build-native.sh
```

产物在 `artifacts/opp/vendors/custom_transformer` 和 `artifacts/qwen36_int4_custom.so`。SDK 和驱动不随仓库分发。

**3. 启动服务**（TP1、MTP=1、FP16 KV，4K 上下文，首次复现用这组配置）

```bash
MODEL_DIR=/absolute/path/ascend310p-llm/weights \
OPP_DIR=/absolute/path/qwen36-build/artifacts/opp/vendors/custom_transformer \
EXTENSION=/absolute/path/qwen36-build/artifacts/qwen36_int4_custom.so \
TP=1 MTP=1 MAX_MODEL_LEN=4096 MEMORY_UTILIZATION=0.90 PORT=18015 \
bash scripts/run-service.sh
```

`MTP=0` 关闭推测解码，`TP=2` 用两颗芯片。长上下文需要单独做内存和质量验证。

**4. 验证生成**

```bash
curl http://127.0.0.1:18015/health
python3 tools/bench_generation_sanity.py \
  --base-url http://127.0.0.1:18015 --model qwen36-w4a8 \
  --out /tmp/qwen36-generation.json --max-tokens 128
```

C8 是独立后端，走 `experimental/run-c8-service.sh`，需要额外构建 C8 库，不能直接给上面的标准启动器加 C8 参数。

## 已知限制

- Graph 路径是实验性的 updatable-graph 开关，不是上游官方默认实现。
- TP2、多模态、长上下文完整精度评测、正式并发/P95 均未验证。
- 公开的 `build-native.sh` 未在全新下载环境端到端逐字验收过（分步构建已验证，见[发布校验记录](https://github.com/ychy7001/ascend310p-llm/blob/ab1a370/docs/qwen36/release-validation.md)）。
- 2026-10-12 完整上传 64 个文件，0 失败，远端 SHA256 全部匹配本地；之后只重新下载过索引和配置文件做字节比对，没有重新下载全部权重再跑一次推理。
- 当前默认部署档是 C16 + MTP2 + Decode Graph + MindIEQ 长 Prefill 分流（这份模型卡测的是早期 C8 配置），最新状态见[仓库 README](https://github.com/ychy7001/ascend310p-llm/blob/ab1a370/README.md)。

## 来源与许可

Qwen3.6 源模型以 Apache-2.0 发布，见 [官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md)。配套代码基于 OpenSensor / matteius 及 vLLM 贡献者的 310P W4 方案，增加了 Qwen3.6 适配、ModelSlim 量化转换和 MTP 支持，详见仓库 [NOTICE](https://github.com/ychy7001/ascend310p-llm/blob/ab1a370/NOTICE)。本项目不是华为、Qwen 或 vLLM 官方发布。
