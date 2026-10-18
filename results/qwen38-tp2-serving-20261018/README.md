# Qwen3.8 Flash Next：单 DUO / TP2 完整运行时验证

2026-10-18。**完整运行时已构建，TP2 GDN 正确性门禁通过；整模加载与真实生成仍在进行，暂未得到整模速度、最大上下文或产品可用结论。** 本目录接续 [上一阶段真实专家 / QSA 验证](../qwen38-tp2-validation-20261018/README.md)。没有改动 Qwen3.6 发布服务代码，也没有触碰本机 RTX 3090 Strata 服务。

## 固定输入与实际环境

| 项目 | 固定值 |
|---|---|
| 设备 | ychy-a2，一张 300I DUO，两颗 310P3，约 44,277 / 43,693 MiB 可见设备内存 |
| 检查点 | [matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i/tree/d32978ccdde1729caec0034571b03a14779698e9) |
| vLLM fork | [opensensor/vllm@3ab5dda29](https://github.com/opensensor/vllm/tree/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5) |
| Ascend fork | [opensensor/vllm-ascend@ffa980c1](https://github.com/opensensor/vllm-ascend/tree/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032) + 本目录验证的两文件补丁 |
| Catlass | 41bf90da655bba3c66d0acd7e00abe33960ecfd6，与 fork 的 .gitmodules 一致，复用 a2 已有副本 |
| 基础镜像 | quay.io/ascend/vllm-ascend:v0.23.0.post1-310p |
| 核心依赖 | CANN 9.1.0，torch 2.10.0+cpu，torch-npu 2.10.0.post4，transformers 5.14.1 |
| 隔离镜像 | localhost/qwen38-runtime:ffa980c-3ab5dda-20261018，仍需挂载 /experiment 源码和算子包，非独立交付镜像 |

`0.30.0+qwen38.*` 是本次 tarball 安装手动指定的包版本标识；源码身份以固定 SHA 为准，不表示官方 v0.30.0 的兼容认证。采用 `VLLM_TARGET_DEVICE=empty` / `--no-deps --no-build-isolation` 复用现有 torch/CANN；本轮未重建 NPU 依赖。

完整 24 个 310P 算子与匹配的 `vllm_ascend_C` 桥接库编译成功，包括 recurrent GDN、chunk GDN、QSA、PLE 与 G128 W4A8。CLI `serve --help` 和桥接库导入通过。版本见 `runtime-*-freeze.txt`，构建日志保留为 `.log.gz`。

环境检查仍有未满足的包声明，详见 `runtime-pip-check.txt`。例如 Ascend fork 声明 FastAPI <0.124，而匹配的 vLLM fork 要求 >=0.133；本轮选择 vLLM 的 FastAPI 0.136.3 / Starlette 1.8.0。310P 不安装 Triton；未使用的 profiler / 视觉依赖也有历史冲突。**不能把当前组件通过等同于所有依赖组合均已验证；HTTP 服务和整模生成是后续门禁。**

## 本轮发现及修复：GDN Prefill 因果掩码错误

使用实际 TP2 局部几何：HK=8、HV=24、DK=DV=128、FP32 recurrent state。通过两个进程同时执行 HCCL all-reduce，结果应为 1+2=3，已通过。

原始社区算子包中：Decode 的 T=1/2/3/4/8 和 Prefill T=32 通过，但 Prefill T=64/70/128/256 在 token 索引 56 出现超门限输出。两个 rank 均能复现，固定输入重复时错误元素数量会变化。

独立复现不需要模型或 WY 计算：Q/K 共享一个单位坐标，使 QK=1；V 为 identity basis；H=0、g=0。`chunk_fwd_o_vllm` 输出的前64列应严格等于 64×64 下三角 0/1 矩阵。未修改的社区 OPP 在两个 rank 各重复3次：

| Rank | 三次错误元素数 | 错误行 |
|---|---|---|
| 0 | 13 / 27 / 30 | 56 |
| 1 | 21 / 23 / 28 | 56 |

这证明是可观测的因果掩码错误，不能解释为正常 FP16 舍入。GDN 状态本身在这些用例中的误差仍较小，因此只检查 final state 会漏掉问题。

定位发现 `InitCausalMask()` 与64-token epilogue 分别生成掩码；只修改前者不能修复后者。最终补丁 [gdn-causal-mask-vector.patch](../../patches/qwen38/gdn-causal-mask-vector.patch) 只修改两个 Ascend C 文件：

1. 以向量操作完整初始化 64×64 掩码，短行使用 lane mask，保留缓冲区切换屏障。
2. 64-token epilogue 直接乘预先初始化的掩码，去掉其中重复生成上三角零值的短 Scalar `SetValue`。

“短标量写与向量流水线交错”是源码和对照支持的故障解释；本实验未进行底层指令跟踪。未修改 W4 数学、GDN 接受计数协议或精度门限。初期仅添加同步、仅修改初始化等失败候选保存在 a2；`failed-scalar-sync.patch` 是失败记录，不应应用到部署版本。

修复前后相对 L2 误差：

| Prefill T | Rank0 原始 | Rank0 修复 | Rank1 原始 | Rank1 修复 |
|---|---:|---:|---:|---:|
| 32 | 0.000448 | 0.000448 | 0.000450 | 0.000450 |
| 64 | 0.049745 | 0.000458 | 0.049599 | 0.000457 |
| 70 | 0.050862 | 0.000456 | 0.032159 | 0.000458 |
| 128 | 0.023383 | 0.000462 | 0.019197 | 0.000459 |
| 256 | 0.023539 | 0.000465 | 0.020019 | 0.000464 |

修复后两个 rank 共20个 GDN 用例通过原 `rtol=3e-3, atol=2e-3` 门限；Prefill 最大状态误差 <0.000755。独立掩码重复6次均为 **0错误 / 最大绝对误差0**。原始数据在 `baseline/`，修复数据在本目录同名 JSON。

另做单一 `chunk_fwd_o` kernel 的首轮计时：HK8/HV24/D128，10次 warmup、50次设备事件计时，保留全部原始样本。

| T | 原始中位延迟 ms | 修复中位延迟 ms | 比值 原始/修复 |
|---|---:|---:|---:|
| 64 | 0.1271 | 0.1227 | 1.036 |
| 128 | 0.1903 | 0.1827 | 1.042 |
| 256 | 0.3750 | 0.3590 | 1.044 |

这是合成输入、单输出 kernel 的局部计时，排除了 WY、专家、PLE、通信等整模成本；不能推导整模 Prefill 提升比例。原始实现有正确性错误，保留它的时间仅用于观察补丁是否引入明显性能回退。

## 权重与整模验证

完整清单见 `checkpoint-manifest.json`：1626文件，共181,713,686,040字节（约169.23 GiB），包含权重、tokenizer、索引和元数据。a2 通过 `hf-mirror.com` 直接断点下载，不经本机传输；LFS 文件核对 SHA256，其余文件核对 Git blob SHA1，全部完成才生成 `download-verified.json`。`.part` 文件绝不视为完整权重。

注意模型文件名中的 W4A16 是存储检查点命名。运行时必须显式设置 `cube_310_int4_a8` 和 `activation_quantization=int8_per_group`；检查点默认 `eager_dequant` 会展开专家，无法按本方案装入一张 DUO。Qwen3.6 的 per-channel OPP 不适用本模型的 G128 协议。

初始整模配置固定为：**TP2 / EP2、原生 W4A8 routed experts、其余 FP16、C16、MTP=0、eager、4K、batch256、maxseq1、memory utilization0.95、纯文本**。不复用社区 TP4 的固定 KV 预算。PLE 继续使用社区按需 mmap，必须保留真实全量 PLE 文件；不把上一阶段仅六行真实数据的 sparse fixture 当整模权重。

检查点完成且GDN门禁通过后，持久化任务自动尝试启动18038并执行3条真实HTTP生成 smoke。smoke 只能验证基本输出与简单指令，不代表系统精度评估，也不是 Decode benchmark。失败会写 `serving-status.json` 和容器日志，便于继续定位。

## a2 资源与复现

```text
工作目录 /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018
模型目录 /home/ychy/models/Qwen3.8-Flash-Next-W4A16-G128-300i-d32978c
社区源码 工作目录/community、工作目录/vllm-fork
原始OPP 工作目录/opp-baseline
修复OPP 工作目录/community/vllm_ascend/_cann_ops_custom
CPU构建容器 qwen38-runtime-build
整模服务容器 qwen38-tp2-text-4k（仅在校验通过后创建）
```

本仓库脚本应复制到上述工作目录；源码来自固定 SHA。按已有 a2 环境复现关键步骤：

```bash
work=/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018
# CPU容器挂载 work:/experiment；先关闭基础镜像中不适用310P的triton目录。
podman exec qwen38-runtime-build bash /experiment/bootstrap-runtime.sh
# catlass/include需来自上述固定版本；第三方CANN构建缓存复用a2已有副本。
# 应用补丁前先保留原始OPP，用于baseline对照。
cd "$work/community"
patch -p1 < "$work/gdn-causal-mask-epilogue.patch"
podman exec qwen38-runtime-build bash -lc \
 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; bash /experiment/community/csrc/build_aclnn.sh /experiment/community ascend310p3'
podman exec qwen38-runtime-build bash /experiment/build-runtime-bridge.sh
# 安装后的Python依赖需保存在镜像，源码及OPP留在挂载目录。
podman commit qwen38-runtime-build localhost/qwen38-runtime:ffa980c-3ab5dda-20261018
bash "$work/run-gdn-tp2-on-a2.sh"
# 原始包对照（预期数值门禁失败；结果写独立目录）：
QWEN38_OPP=/experiment/opp-baseline/vendors/custom_transformer \
QWEN38_PROBE_OUTPUT=/experiment/gdn-baseline \
bash "$work/run-gdn-tp2-on-a2.sh"
```

不要对已打补丁的源码重复应用。`run-gdn-tp2-on-a2.sh` 会停止 a2 原 Qwen3.6 服务；用户已授权两颗芯片专用于实验。原服务 inspect 恢复快照留在 a2，未公开完整环境配置。

```bash
# 下载（本轮已在后台运行，不要重复启动）：
python3 "$work/qwen38_download_checkpoint.py" \
 --manifest "$work/qwen38-api-blobs.json" \
 --output /home/ychy/models/Qwen3.8-Flash-Next-W4A16-G128-300i-d32978c \
 --workers 48
# 本轮自动流程已运行；以下用于观察：
cat "$work/serving-status.json"
tail -n 20 "$work/download-full.log"
tail -n 20 "$work/serving-orchestration.log"
podman logs --tail 50 qwen38-tp2-text-4k
# 如需在自动任务未运行时手动启动，先确认没有同名容器：
bash "$work/launch-text-tp2-on-a2.sh"
python3 "$work/qwen38_serving_smoke.py" --output "$work/serving-smoke.json"
```

`launch-text-tp2-on-a2.sh` 检查完整下载清单、两个 rank 的20个GDN用例和精确掩码结果。服务端完整参数见 `serve-text-tp2.sh`。下一步是观察真实权重加载峰值、首次生成，再决定是否扩展到 MTP、Graph 与长上下文。
