# Qwen3.8 Flash Next：a2 双 310P 最小验证

日期：2026-10-18。状态：**真实权重专家算子、融合专家 MLP 和 TP2 局部 QSA 通过；整模 TP2 服务尚未验证。**

这次验证选择社区均匀 INT4 检查点和 OpenSensor 运行时，不直接移植本机 Strata 的 IQ3_XXS GGUF 执行器。两种格式不是同一编码或同一精度方案，不能混用权重。

## 固定输入与环境

- [社区权重](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i/tree/d32978ccdde1729caec0034571b03a14779698e9)，revision `d32978ccdde1729caec0034571b03a14779698e9`。
- [OpenSensor Ascend 源码](https://github.com/opensensor/vllm-ascend/tree/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032)，此次原样编译其中四个 W4 算子和 `qsa_sparse_attention_v310`。Torch adapter 使用同一快照，通过独立探针命名空间加载。
- 模型卡指定的 [vLLM fork](https://github.com/opensensor/vllm/tree/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5)；**本次未安装/启动该完整运行时**。
- a2：单张 DUO、两颗 310P3；`npu-smi 26.1.1`。本次 NPU 测试仅使用 `/dev/davinci1`。
- 编译/探针镜像：`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，CANN 9.1.0，torch `2.10.0+cpu`、torch-npu `2.10.0.post4`、vLLM `0.23.0+empty`、vLLM-Ascend `0.23.0.post1`。它只是探针环境，不能据此宣称该旧版已经支持 Qwen4Exp 整模。
- 探针扩展 SHA256：`6bdef9a1cbda83a88a5f4e82fa0e61a397b1c35814bcfc2294bb4e83ab5a2f32`。
- 自定义 OPP `libcust_opapi.so` SHA256：`a76163318166a69e76ab6db2b32956ecbac1db769dbfefe1b8f68d775e554943`。

## 1. 容量核算：有继续尝试 TP2 的价值，尚非实测峰值

索引共 222,746 个张量、1,610 个 safetensors 文件。`capacity.json` 使用固定 revision 的文件大小和源码中的原生权重布局计算：

| 项目 | 大小 |
|---|---:|
| 原生 W4 专家，整个模型 | 61.5234 GiB |
| TP2/EP2 每 rank 专家 | 30.7617 GiB |
| 主模型其余浮点文件总量，含文件头 | 9.2160 GiB |
| 每 rank 权重保守上界：除专家外全部复制 | 39.9777 GiB |
| 每 rank 权重保守上界：再计入词嵌入/LM head 的 TP2 分片 | 38.7937 GiB |
| PLE 磁盘表 | 95.3679 GiB |
| 独立 MTP 浮点文件 | 4.8562 GiB |
| 视觉浮点文件 | 0.8363 GiB |

专家配置：48 层、512 experts/layer、top-10、hidden=2560、intermediate=640。Gate/Up 为 `[640,2560]`，Down 为 `[2560,640]`。每 rank 持有 256 个完整专家，未把每个专家再做列切分。

原生格式每个权重占 `0.5 + 6/128` 字节：INT4 codes、FP16 scale、FP16 offset、FP16 weight_sum。检查点 offset 是 INT8；加载时转 FP16 并增加 weight_sum，所以不能只把磁盘专家大小除以二当作显存需求。

上界还没有扣除源码支持的 GDN/QSA 投影及 shared experts 分片，因而偏保守；但也**没有加入 KV/QSA/Indexer cache、GDN/PLE 状态、工作区、图池、碎片和加载瞬时峰值**。单芯片总显存约 42–43 GiB，剩余空间有限。这份原生 W4 格式仅专家就占 61.52 GiB，因此 TP1 无法直接装入单颗 48 GB 芯片。无法据此承诺长上下文、MTP 或服务启动成功。

PLE 使用 lazy mmap 时不会一次把 95 GiB 表装入 32 GiB 主机内存，但所访问页面会进入 OS page cache。源码 `physical_bytes=0` 不是主机真实内存开销为零，也不代表冷读延迟为零。

## 2. 真实权重和 310P 执行结果

通过校验 `HTTP 206 Content-Range` 分段获取 layer 0 的 expert 0、256 的全部三个投影（共 18 个张量），另取 PLE shard 0、127 各三行，实际下载 **5,783,376 字节**。expert 0/256 分属 TP2 的不同专家范围；本次合并为两专家小 bank 在一颗 NPU 上验证计算，没有运行 HCCL/EP 通信。

| 验证 | 结果 | 范围 |
|---|---|---|
| 原社区 lazy PLE 实现读真实行 | PASS | 六行，含 shard 首尾、乱序与重复读取 |
| 社区 QSA/Indexer/PLE 主机测试 | 88 PASS / 1 SKIP | 原测试，跳过的是需要完整本地检查点的测试 |
| QSA TP2 头切分 CPU 对照 | PASS | 24Q/2KV/256 切成每 rank 12Q/1KV/256 |
| 真实专家单投影 310P | 18/18 PASS | Gate/Up/Down × 输入 token 1/2/4/8/128/512 |
| 融合 Gate+Up → SwiGLU pack → Down | 4/4 PASS | token 1/2/4/8，两条合成路由/token；每例重复 10 次逐位一致 |
| TP2 局部 QSA 原生 310P 算子 | 4/4 PASS | token 1/2/8/128，12Q/1KV/256，非连续 NZ 页、67 个被选 token |

单投影相对相同 W4A8 量化规则的 FP32 数学参考，relative L2 约 `0.000203–0.000216`，即 **0.0203%–0.0216%**。相对同一 W4 权重的 A16 激活参考约 **0.60%–0.68%**；这只衡量本次合成输入的激活量化影响，未测原始 BF16→W4 权重量化误差或任务准确率。

融合 MLP 的 relative L2 约 `5.0e-7–1.15e-5`。QSA relative L2 约 `0.000268–0.000275`，最大绝对差 `0.00024414`。

所有 NPU 路径直接调用编译的自定义算子，未设置 fallback。MatMul 前只有测试用的 CPU 参考反量化；被测 NPU 路径保留打包权重。输入 hidden states、路由、Q/K/V 和 sparse selections 为合成数据，**不构成完整生成或模型质量验证**。

### 一个已排除的适配陷阱

最初试用现有 Qwen3.6 专用 per-channel OPP，Gate/Up 误差很大。其激活 pack 和矩阵优化针对对称、按通道 scale 的 Qwen3.6 权重，不能处理该非对称 G128 检查点。切换为上述固定社区源码重新编译的 G128 OPP 后通过。

失败原始数据保留在 `qwen36-per-channel-package-incompatible.json`；它是**不匹配包的诊断结果**，不能作为社区原始实现的错误或性能结论。现有 Qwen3.6 代码/OPP 均未替换。

### 探针时间的解读

`npu-experts.json` 保留每例 3 组、每组 20 次调用的计时，预热 5 次。时间包含 host dispatch、激活 pack、matmul 和组末同步，未隔离纯 kernel latency，未测试 Graph。两个专家的小 bank 具有与真实 512-expert 路由不同的缓存行为。

例如 batch=1，两条路由时，单投影 pack+matmul 约 31–62 μs。**不能把这些数值直接换算为整模 tok/s，也不能据此与本机 Strata 或 Qwen3.6 服务比较。**

## 3. 服务器资源与复现

本机代码仓库：`/home/ychy/workspace/ascend310p-qwen36`。
a2 独立实验目录：`/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018`，包含源码快照、fixture、编译产物和原始日志。

- `community/`：固定的 OpenSensor Ascend 源码；`vllm-fork/`：匹配的 vLLM 源码（已准备，未安装）。
- `grouped-opp/vendors/custom_transformer/`：新编译的分组算子包。
- `probe-build/qwen36_int4_custom.so`：独立 Torch adapter，包含 W4 和 QSA 探针入口。
- `experts.safetensors`：仅两个专家、18 张量的真实小样本，**不是完整检查点**。
- 本机 `reference/qwen38-310p-fixtures/float-*.safetensors` 是稀疏 PLE fixture，只有 manifest 记录的行是真实数据；其余是零空洞，**严禁作为完整模型发布或部署**。

小样本获取：

```bash
python3 tools/qwen38_fetch_fixture.py \
  --index /tmp/qwen38-index.json \
  --output /home/ychy/workspace/reference/qwen38-310p-fixtures
python3 tools/qwen38_capacity_audit.py \
  --index /tmp/qwen38-index.json --blobs /tmp/qwen38-api-blobs.json \
  --output results/qwen38-tp2-validation-20261018/capacity.json
```

`source-manifest.json` 记录输入 SHA256。索引来自上述固定 HF revision 的 `model.safetensors.index.json`；文件大小列表来自 `/api/models/<repo>/revision/<revision>?blobs=true`。下载脚本默认本机代理 `1081`，每个 Range 严格校验服务器响应，缓存后可重跑。

CPU：在带 torch/numpy/safetensors/pytest 的隔离环境运行：

```bash
python3 tools/qwen38_verify_fixture.py --fixture <fixture目录> \
  --source <固定Ascend源码目录> --output ple-cpu.json
python3 tools/qwen38_host_gates.py --source <固定Ascend源码目录> --junit host-tests.xml
```

`build-on-a2.sh`、`run-on-a2.sh`、`run-mlp-on-a2.sh` 保留此次服务器路径和完整命令。它们是实验记录，包含停止/按快照恢复 chip1 服务的操作，重跑前检查容器名、目录和当前占用。构建不挂载 NPU。首次编译第三方 protobuf 较慢；已使用 `CMAKE_BUILD_PARALLEL_LEVEL=4` 并保留缓存。

## 4. 下一阶段门禁

1. **先冻结整套运行时**：使用匹配的 OpenSensor vLLM/Ascend fork、其 requirements（例如 transformers 5.14.1）、模型卡要求的 GDN FP32/QSA 接口和完整 OPP。当前旧版 vLLM 0.23 的单算子通过，不等于整模运行时兼容。不要只拷几个模型 Python 文件进现有 Qwen3.6 镜像。
2. **再准备完整权重**：直接在 a2 下载固定 revision，完整核验文件，保持原始索引和元数据；此次小样本不能启动服务。确认 PLE lazy、EP 过滤在真正的 checkpoint loader 中生效，控制主机峰值。
3. **最小整模启动**：TP2/EP2、纯文本、MTP=0、单请求、4K 上下文、较小 prefill chunk、先 eager/C16。显式选择 `cube_310_int4_a8`，避免检查点默认 `eager_dequant` 造成意外浮点权重展开。必须记录实际每 rank 权重峰值、剩余缓存及冷 PLE I/O。
4. **正确性后再优化**：完成连贯生成与固定质量样本，验证 HCCL、GDN 实际生产头形状和状态；再开 Decode Graph、尝试 MTP，逐级扩大上下文。双芯片速度、长上下文和本机 Strata 的差异全部以届时端到端数据为准。

目前结论：**核心算子和 TP2 局部 QSA 的硬件阻塞已经排除，容量估算支持继续尝试；尚不能宣称 a2 双芯片已经完整运行 Qwen3.8，也不能承诺最大上下文和整模速度。**
