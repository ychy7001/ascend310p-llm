# Qwen3.8 Flash Next：单 DUO / TP2 完整运行时验证

2026-10-18。**完整Qwen3.8 Flash Next已在a2一张300I DUO / 两颗310P3上运行，3条真实生成和4096-token实际请求通过。512/128在最终配置下重复3次Decode13.82–13.89tok/s；尚无MTP、Graph、更长上下文或产品质量结论。** 本目录接续 [上一阶段真实专家 / QSA 验证](../qwen38-tp2-validation-20261018/README.md)。没有改动 Qwen3.6 发布服务代码，也没有触碰本机 RTX 3090 Strata 服务。

## 固定输入与实际环境

| 项目 | 固定值 |
|---|---|
| 设备 | ychy-a2，一张 300I DUO，两颗 310P3，约 44,277 / 43,693 MiB 可见设备内存 |
| 检查点 | [matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i/tree/d32978ccdde1729caec0034571b03a14779698e9) |
| vLLM fork | [opensensor/vllm@3ab5dda29](https://github.com/opensensor/vllm/tree/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5) |
| Ascend fork | [opensensor/vllm-ascend@ffa980c1](https://github.com/opensensor/vllm-ascend/tree/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032) + 本目录的3个独立补丁（GDN掩码、通信缓冲、单请求预算） |
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

完整清单见 `checkpoint-manifest.json`：1626文件，共181,713,686,040字节（约169.23 GiB），包含权重、tokenizer、索引和元数据。a2 通过 `hf-mirror.com` 直接断点下载，不经本机传输；LFS 文件核对 SHA256，其余文件核对 Git blob SHA1，全部完成才生成 `download-verified.json`。`.part` 文件绝不视为完整权重。下载器可选 `--range-chunk-mib 64 --range-workers 8` 为剩余大分片启用有界并行 Range：严格核对 206 与 Content-Range、按顺序合并临时片段，再核对完整文件哈希。

注意模型文件名中的 W4A16 是存储检查点命名。运行时必须显式设置 `cube_310_int4_a8` 和 `activation_quantization=int8_per_group`；检查点默认 `eager_dequant` 会展开专家，无法按本方案装入一张 DUO。Qwen3.6 的 per-channel OPP 不适用本模型的 G128 协议。

初始整模配置为：**TP2 / EP2、原生 W4A8 routed experts、其余 FP16、C16、MTP=0、eager、4K、batch256、maxseq1、memory utilization0.95、纯文本**。不复用社区 TP4 的固定 KV 预算。PLE 继续使用社区按需 mmap，必须保留真实全量 PLE 文件；不把上一阶段仅六行真实数据的 sparse fixture 当整模权重。

2026-10-18 09:51 UTC，完整1626文件已校验通过（`checkpoint-download-verified.json`）。日志确认原生后端，每rank加载256/512专家，权重37.2647GiB。以下保留实际排障顺序及失败证据，避免将装载成功误认为服务可用。

| 阶段 | 观测与处理 |
|---|---|
| 首次启动 | OPP `set_env.bash` 与 `set -u` 冲突；改为显式OPP路径及动态库路径。 |
| 权重完整装载 | 约189–190秒；随后profile的HCCL申请420,478,976字节失败，见 `serving-attempt2-hccl-oom.log.gz`。 |
| 通信缓冲 | 单设HCCL_BUFFSIZE仍失败；fork的PG options固定200MiB。可选实验patch设置PG=64MiB，profile通过。 |
| Prefix Cache | 最小64个Mamba状态槽预留3,694,657,536字节/rank，使KV预算为0；见 `serving-attempt4-prefix-pool-budget.log.gz`。首轮关闭prefix，仅留当前请求状态。 |
| 首次真实生成 | 3条基本生成通过，512/1024输入可完成；自动KV约0.97GiB，但3968输入未获调度。worker栈为等待RPC，见 `4k-engine-stack.txt`。 |
| 显式2GiB试验 | 跳过profile后，首次生成的HCCL 135,266,304字节申请OOM；见 `serving-attempt6-fixedkv-oom.log.gz`。该参数已撤回。 |
| 单请求预算修复 | 单请求没有live lane映射对象，旧规划遗漏物理→planner换算。补丁将其纳入既有compact live-state预算路径；CPU探针3例通过，恢复自动profile后服务再次健康、3条生成再次通过。 |

补丁分别为 [hccl-buffer-experiment.patch](../../patches/qwen38/hccl-buffer-experiment.patch) 和 [compact-single-request-budget.patch](../../patches/qwen38/compact-single-request-budget.patch)。前者未设置实验变量时保留200MiB默认；后者只补齐compact单请求分支，不改变GDN数值算法。`compact-budget-before/after.json` 使用实际FullAttentionSpec/MambaSpec做独立预算检查：原始单请求planner预算123450，而物理分配对应的正确预算为1231040；多请求与prefix对照本来通过，修复后3例均通过。

修复后的实际规划（rank1）：单个递推状态池57,729,024字节，物理attention预算977,449,830字节，578块；换算后的逻辑预算约10.47GiB。它不是物理NPU新增10GiB。Torch allocated由约37.40GiB升至38.23GiB，仍在设备容量内。框架报告67,642-token容量，本轮仍只验证4096上限，不能据此承诺67K长上下文。

持久化任务在完整下载和GDN门禁通过后启动18038、执行基本HTTP smoke；失败保留状态和日志。完整源码/算子包、真实PLE文件均留a2。PLE是按需mmap读取的CPU/磁盘表，不代表整个169GiB检查点全部驻留NPU。

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
patch -p1 < "$work/hccl-buffer-experiment.patch"
patch -p1 < "$work/compact-single-request-budget.patch"
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
 --workers 8 --range-chunk-mib 64 --range-workers 8
# 本轮自动流程已运行；以下用于观察：
cat "$work/serving-status.json"
tail -n 20 "$work/download-full.log"
tail -n 20 "$work/serving-orchestration.log"
podman logs --tail 50 qwen38-tp2-text-4k
# 如需在自动任务未运行时手动启动，先确认没有同名容器：
bash "$work/launch-text-tp2-on-a2.sh"
python3 "$work/qwen38_serving_smoke.py" --output "$work/serving-smoke.json"
```

`launch-text-tp2-on-a2.sh` 检查完整下载清单、两个 rank 的20个GDN用例和精确掩码结果。服务端完整参数见 `serve-text-tp2.sh`。当前可用参数已关闭Prefix Cache并使用64 MiB通信组缓冲；后续扩展MTP、Graph、长上下文须分别验证。

启动期间 `qwen38_watch_resources.py` 每5秒记录主机可用内存 / Swap 和原始 `npu-smi info`，结果在 a2 的 `serving-resources.jsonl`。它提供采样观察的最大占用，不代表捕获所有瞬时峰值。

## 完整检查点基础生成与首轮性能

最终服务在2026-10-18 10:31 UTC完成3条真实生成：17+25输出42，中文内存解释连贯，固定词输出READY，见 `serving-smoke.json`。这是基本输出检查，不是模型级质量评测。

最终配置：TP2/EP2、G128原生W4A8 routed experts、非专家FP16、C16、MTP0、eager、maxlen4096、batch256、maxseq1、memory utilization0.95、纯文本；**Prefix Cache关闭 / mamba cache mode none**；`QWEN38_HCCL_BUFFER_MIB=64`；自动profile，**没有显式KV cache memory参数**。源码需3个补丁，参数见 `serve-text-tp2.sh`。最终装载190.27秒，engine初始化11.56秒，完整日志 `serving-final.log.gz`。

`bench_http_stream.py` 使用固定token输入、temperature0、ignore_eos、SSE usage核对。每组先做32输出token warmup，再计时；所有请求串行、Prefix Cache关闭。Decode=(output_tokens-1)/(最后usage时间-首个非空文本事件时间)，端到端Prompt吞吐=prompt_tokens/TTFT；后者含HTTP/调度/首次Decode等开销，不能称为纯Prefill kernel吞吐。原始逐事件数据保留。

| 最终配置 输入/输出 | 计时重复数 | TTFT 秒 | Decode tok/s | 输入/TTFT tok/s |
|---|---:|---:|---:|---:|
| 512 / 128 | 3 | 1.909–1.911 | 13.82–13.89 | 267.9–268.1 |
| 3968 / 128 | 1 | 15.841 | 12.68 | 250.5 |

3968/128的warmup和计时请求均完成；计时usage明确为3968输入、128输出、共4096token（`serving-bench-compact-3968-128.jsonl`）。这是实际完成的4K请求，不是只设置了max_model_len。它不证明长上下文检索质量或最大容量。单次4K计时也不足以给出稳定P95。

历史自动KV约0.97GiB版本的结果保留在 `serving-bench-512-128.jsonl` 和 `serving-bench-1024-64.jsonl`：512/128三次Decode13.76–13.95tok/s；1024/64一次TTFT3.765s、Decode13.83tok/s。最终修复后的同条件512复测没有明显速度回退。

`resource-summary.json` 统计最终实例79次采样，原始 `serving-resources.jsonl.gz` 同时包含前期失败对照。以下是采样观察到的最大占用，不是精确瞬时峰值：

| 设备 | 最大占用 MiB | 可见总量 MiB | 最少空闲 MiB |
|---|---:|---:|---:|
| 0 | 42,282 | 44,277 | 1,995 |
| 1 | 41,746 | 43,693 | 1,947 |

本轮没有修改lm_head/PLE projection精度。专家本身61.52GiB，TP1不能装进一颗48G；TP2已经完成实际文本推理。PLE约95GiB在磁盘/主机按需mmap，其访问模式、页缓存命中会影响性能，不能把短重复文本数据推导为全部业务场景性能。

复现计时（不要并发运行其他请求）：

```bash
python3 "$work/bench_http_stream.py" --base-url http://127.0.0.1:18038 \
 --model qwen38-w4a8-tp2-text --prompt-tokens 512 --output-tokens 128 \
 --repeats 3 --out "$work/serving-bench-compact-512-128-new.jsonl"
python3 "$work/bench_http_stream.py" --base-url http://127.0.0.1:18038 \
 --model qwen38-w4a8-tp2-text --prompt-tokens 3968 --output-tokens 128 \
 --repeats 1 --out "$work/serving-bench-compact-3968-128-new.jsonl"
```

a2端口18038保留运行，模型名 `qwen38-w4a8-tp2-text`。本机3090 Strata服务未改动；本方案使用社区均匀INT4检查点，未把IQ3_XXS GGUF直接交给NPU加载。

## 工程判断与后续边界

单DUO/TP2文本推理已从容量估算推进到真实生成，值得继续，但当前只是实验基线。优先冻结这个4K配置及数值补丁，再独立评估Decode Graph的收益与捕获内存；当前最少只剩约1.9GiB设备余量。随后才扩大上下文并做检索/生成质量评估。

MTP、C8、Prefix Cache恢复、视觉及多并发均未在此完整模型配置上验证。不能直接套用Qwen3.6的成熟参数，也不能将框架报告的67,642-token容量当成产品承诺。需要减内存时可评估社区已有lm_head/PLE projection动态W8方案，但应另设质量与速度对照，不能混入本轮FP16基线。
