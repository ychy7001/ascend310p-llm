# Qwen3.8-Flash-Next：一张 300I DUO 的 TP2/EP2 bring-up

## 1. 结论先行

完整社区检查点已在 a2 的一张 300I DUO、两颗 310P3 上完成文本推理。路径是：原生 G128 W4A8 专家按 EP2 分布，其他层按 TP2 执行，约 95 GiB PLE 留在磁盘按需 mmap；修复启动与预算问题后，用可中断 ACL Decode Graph 提速。当前产品候选为 **16K / batch256 / C16 / MTP0 / Decode Graph / 单请求 / Prefix Cache 关闭**。

| 能力 | 实测结果 / 状态 |
|---|---|
| 完整权重与 4K eager | 每 rank 权重 37.2647 GiB；512/128 Decode 13.82–13.89 tok/s |
| 4K Decode Graph | 512/128 为 16.50–16.72 tok/s，中位 +20.14%；3968/128 中位 +28.55%（初步） |
| 16K Graph | 三次完成 16256+128=16384；Decode 中位 15.59 tok/s，TTFT 69.184–69.299 s |
| 长请求状态与检索 | 16256-token 中部口令命中；交替长短请求 8/8 精确匹配 |
| 32K | 可启动，32256+128 请求无法调度，当前不可用 |
| batch512 | 8K Prefill 有效改善约 11.3%，16K 请求无法调度，不采用 |
| MTP / C8 / 并发 / 多模态 | 未验证 |

全部实验日期为 2026-10-18。下文保留原始记录入口；这是文本单流实验配置，完整质量评估仍待完成。

## 2. 为什么是 TP2：先把容量算对

要解决的问题不是模型文件能否下载，而是原生权重布局能否放进两颗芯片。[社区容量调查](../../results/qwen38/tp2-feasibility-20261018/README.md)确认模型卡实测为两张 DUO、四颗 310P3 的 TP4/EP4。选择从社区缩到 TP2，是因为已有 Qwen4Exp、G128 算子、QSA/GDN 和 PLE lazy 路径；直接移植 Strata 还需要重做 Ascend 执行器，IQ3_XXS GGUF 也不是均匀 INT4 编码。

[固定检查点容量审计](../../results/qwen38/tp2-validation-20261018/README.md)给出如下布局：

| 项目 | 容量 / 布局 |
|---|---|
| 原生 W4 专家全量 | 61.5234 GiB，单颗 48 GB 装不下 |
| EP2 每 rank 专家 | 30.7617 GiB；每 rank 持有 256/512 个完整专家 |
| 其余主模型浮点文件，含文件头 | 9.2160 GiB |
| 每 rank 保守权重上界，其他层全复制 | 39.9777 GiB |
| 再计入词嵌入 / LM head 的 TP2 分片 | 38.7937 GiB |
| PLE 磁盘表 | 95.3679 GiB |
| 独立 MTP / 视觉浮点文件 | 4.8562 / 0.8363 GiB |

模型为 48 层、hidden=2560、512 experts/layer、top-10、expert intermediate=640。专家按 EP 分布，每个专家保持完整矩阵。原生格式每权重占 `0.5 + 6/128` 字节：INT4 codes 加 FP16 scale、offset、weight_sum。磁盘 INT8 offset 加载后转 FP16，还会增加 weight_sum；不能直接把磁盘专家大小除以二当设备内存。

PLE 表比 a2 的 31 GiB 主机 RAM 大。社区 `AscendPLELazyShardEmbeddingMethod` 按需 mmap Safetensors 行，不先分配整表 host 内存，并跳过 lazy transport 的整表预算检查，因此选这条路径。真实全量 PLE 文件必须保留，访问页会进入 OS page cache，冷磁盘和多 rank 重复读取会影响延迟。

容量上界尚未扣除 GDN/QSA 投影和 shared experts 的分片，也未加入 cache、状态、工作区、图池和加载瞬时峰值。调查中的双芯片约 12 GiB 算术余量只是估算；最终装载日志为 37.2647 GiB/rank，采样占用见第 4 节。

## 3. 先验证算子，再上整模

先下载 layer 0 的 expert 0、256 的三个投影，以及 PLE shard 0、127 的六行，总计 5,783,376 字节。两个专家覆盖 EP2 两侧范围，先在一颗 NPU 上合为小 bank，排除权重协议与局部算子阻塞。

| 验证问题 | 结果 | 判据 / 覆盖 |
|---|---|---|
| lazy PLE 能否读取真实行 | PASS | 首尾、乱序、重复读取，共六行 |
| QSA / Indexer / PLE 主机逻辑 | 88 PASS / 1 SKIP | 跳过项需要完整本地检查点 |
| QSA 能否切到 TP2 | CPU 对照 PASS | 24Q/2KV/256 → 每 rank 12Q/1KV/256 |
| 真实专家 Gate/Up/Down | 18/18 PASS | token 1/2/4/8/128/512；相对 W4A8 FP32 参考 relative L2 约 0.000203–0.000216 |
| Gate+Up → SwiGLU pack → Down | 4/4 PASS | token 1/2/4/8；relative L2 约 5.0e-7–1.15e-5；每例重复 10 次逐位一致 |
| TP2 局部 QSA 原生算子 | 4/4 PASS | token 1/2/8/128；非连续 NZ 页、67 个被选 token；relative L2 约 0.000268–0.000275 |

最初使用 Qwen3.6 per-channel OPP，Gate/Up 误差很大。该包针对对称、按通道 scale 的权重，无法处理这里的非对称 G128 协议。换成固定社区源码重新编译的 G128 OPP 后通过。

整模还要显式选 `cube_310_int4_a8` 与 `activation_quantization=int8_per_group`。检查点名字虽为 W4A16，其默认 `eager_dequant` 会展开专家为浮点，破坏容量方案。

[完整算子记录](../../results/qwen38/tp2-validation-20261018/README.md)中的 hidden states、路由和 Q/K/V 为合成输入；这些测试没有 EP 通信。相对 A16 激活参考的单投影误差约 0.60%–0.68%，尚未测原始 BF16→W4 权重量化误差或任务准确率。

## 4. 整模启动：逐个解除阻塞

使用匹配的两个 fork 和完整 24 个 310P 算子、`vllm_ascend_C` 桥接库。旧版 vLLM 0.23 探针环境只验证单算子，整模必须安装固定源码。起步选择文本、4K、C16、MTP0、eager、batch256、maxseq1、自动 profile，先让真实请求完成。

| 遇到什么 | 原因 | 怎么解决 / 结果 |
|---|---|---|
| OPP 环境脚本启动失败 | `set_env.bash` 与 `set -u` 冲突 | 显式设置 OPP 路径和动态库路径 |
| 权重装载后 profile OOM，申请 420,478,976 字节失败 | 通信缓冲挤占剩余内存；只设 `HCCL_BUFFSIZE` 无效，fork PG options 固定 200 MiB | 用[通信缓冲补丁](../../patches/qwen38/hccl-buffer-experiment.patch)，设置 PG=64 MiB，profile 通过 |
| KV 预算为 0 | Prefix Cache 至少预留 64 个 Mamba 状态槽，占 3,694,657,536 字节/rank | 关闭 Prefix Cache，`mamba_cache_mode=none`，只留当前请求状态 |
| 短请求能生成，3968 输入不调度 | 自动物理 KV 约 0.97 GiB；单请求没有 live lane 映射，旧规划漏掉物理→planner 换算 | 用[单请求预算补丁](../../patches/qwen38/compact-single-request-budget.patch)，纳入既有 compact live-state 路径；CPU 3 例通过 |
| 显式设 2 GiB KV 后首次生成 OOM | 跳过 profile 后，HCCL 又申请 135,266,304 字节失败 | 撤回显式 KV 参数，恢复自动 profile；修复预算后完成 4K 请求 |

GDN Prefill 因果掩码错误用[向量掩码补丁](../../patches/qwen38/gdn-causal-mask-vector.patch)修复，定位与验证详见 [GDN 因果掩码问题](gdn-causal-mask-bug.md)。

| 最终 eager 运行证据 | 数据 |
|---|---|
| 完整检查点校验 | 1626 文件，181,713,686,040 字节（约 169.23 GiB） |
| 装载 / engine 初始化 | 190.27 / 11.56 秒 |
| 每 rank 权重 | 37.2647 GiB |
| rank1 递推状态池 / 物理 attention 预算 | 57,729,024 / 977,449,830 字节；578 块 |
| chip0 采样最大占用 / 可见总量 / 最少空闲 | 42,282 / 44,277 / 1,995 MiB |
| chip1 采样最大占用 / 可见总量 / 最少空闲 | 41,746 / 43,693 / 1,947 MiB |
| 真实生成 | 3 条通过：17+25 输出 42、中文解释、固定词 READY |

[排障日志与资源原始记录](../../results/qwen38/tp2-serving-20261018/README.md)包含失败尝试。上表设备占用来自 79 次采样，不是精确瞬时峰值。planner 约 10.47 GiB 是换算后的逻辑预算；框架报告的 67,642-token 容量不是可用上下文承诺。依赖声明仍有冲突，固定安装组合见原记录的 freeze 与 pip-check。

## 5. 4K 基线与 Decode Graph：收益在 Decode

在相同权重与服务配置上，移除 `--enforce-eager`，开启 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，使用 `mode=0`、`FULL_DECODE_ONLY`、capture sizes `[1]`。这是可中断 ACL Graph：PLE 主机查表/传输、动态选择的 QSA 操作在图间执行；没有启用 torch.compile。

| 输入/输出 token | eager TTFT | Graph TTFT | eager Decode | Graph Decode | Decode 中位变化 |
|---|---:|---:|---:|---:|---:|
| 512/128，双方各 3 次 | 1.909–1.911 s | 1.904–1.992 s | 13.82–13.89 tok/s | 16.50–16.72 tok/s | +20.14% |
| 3968/128，eager 1 次、Graph 3 次 | 15.841 s | 16.137–16.239 s | 12.68 tok/s | 16.26–16.35 tok/s | +28.55%，初步 |

Prefill 本轮无收益：输入/TTFT 中位数分别从 268.04 降为 258.52 tok/s、从 250.49 降为 245.01 tok/s；TTFT 中位数增加 3.68% / 2.24%。因此保留 Graph 的依据是 Decode 提速。

捕获日志显示一个形状；模式统计有 883 次 `FULL`、96 次 `NONE`，前者均为单 token Decode，较大 Prefill 块为 `NONE`。两 rank 图内存约 0.18 / 0.19 GiB，最小物理 attention 仍为 578 块。三次 3968+128 实际请求完成，基本生成 3/3、交替请求 8/8 通过。

[Graph 原始对照](../../results/qwen38/decode-graph-20261018/README.md)保留文本与 SSE 事件。3968 档输出哈希未一致，512 档两侧哈希集合一致；完整质量等价仍需评估。+28.55% 的 eager 只有一次样本，属于初步结果。

## 6. 往长上下文走：保住 16K，再看 Prefill

先只将 `max_model_len` 从 4096 改成 16384，其他条件固定。每档先预热 32 输出 token，再执行三次计时并校验 usage。

| 输入/输出 token | 次数 | TTFT | Decode 范围 | Decode 中位 | 输入/TTFT 中位 |
|---|---:|---:|---:|---:|---:|
| 8064/128 | 3 | 33.671–33.866 s | 16.17–16.20 tok/s | 16.19 tok/s | 238.66 tok/s |
| 16256/128 | 3 | 69.184–69.299 s | 15.56–15.60 tok/s | 15.59 tok/s | 234.87 tok/s |

[16K 验证](../../results/qwen38/long-context-20261018/README.md)还通过基本生成 3/3、中部唯一口令检索和交替请求 8/8。口令位于 token8120，返回 `ASCEND732941`，实际输出 14 token、总长 16270、耗时 72.19 s；交替检查包含 3033-token 请求后的短请求。日志中最小物理 attention 仍为 578 块。

| 要解决的问题 | 怎么决定 / 实测 | 结果 |
|---|---|---|
| 能否直接提高到 32K | 仅改 `max_model_len=32768`；装载、Graph capture、3 条基本生成成功，物理块仍为 578/584 | 32256+128=32384 请求持续 Waiting=1、Running=0、KV usage=0.0%，没有输出 usage；32K 不可用 |
| batch512 能否减少长 Prefill 轮次 | GDN 与 grouped 投影门禁通过；batch 与 grouped chunk 同时改为 512；8K TTFT 中位约 29.98 s，对照 33.79 s，改善约 11.3% | Decode 约 15.98 对照 16.19 tok/s，下降约 1.3%；16K 请求无法调度，不采用 |
| batch512 为什么丢失容量 | 峰值激活从约 0.20 升至 0.39 GiB，物理 attention 块从 578 降至 504 | 8K Prefill 收益不足以抵偿 16K 容量损失，恢复 batch256 |

[32K 边界](../../results/qwen38/long-context-32k-20261018/README.md)表明瓶颈是物理 KV 容量与调度；提高模型长度不会增加物理 KV。[batch512 对照](../../results/qwen38/prefill-batch512-20261018/README.md)因此也未进入默认配置。产品候选继续为 **16K / batch256 / C16 / MTP0 / Decode Graph**。

这里的计时使用固定重复文本，PLE 页缓存命中可能不同于自然业务输入；单一 canary 和闭式答案只覆盖有限检索与状态检查。

## 7. 复现：以 a2 固定环境为起点

实验使用 **rootless Podman**。Docker 用户需自行调整设备挂载和用户映射，尚未在 Docker 上验证。a2 当前离线，以下命令对应留存环境，尚未在其他主机重跑。

| 项目 | 固定环境 |
|---|---|
| 设备 / 主机 | ychy-a2，单 DUO 双 310P3；主机 RAM 31 GiB |
| 基础镜像 | `quay.io/ascend/vllm-ascend:v0.23.0.post1-310p` |
| 核心依赖 | CANN 9.1.0，torch `2.10.0+cpu`，torch-npu `2.10.0.post4`，transformers `5.14.1` |
| 实验镜像 | `localhost/qwen38-runtime:ffa980c-3ab5dda-20261018`；依赖 `/experiment` 源码与 OPP 挂载 |
| a2 工作目录 | `/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018` |
| a2 模型目录 | `/home/ychy/models/Qwen3.8-Flash-Next-W4A16-G128-300i-d32978c` |

源码分别放在工作目录的 `community/` 与 `vllm-fork/`。构建按[整模记录](../../results/qwen38/tp2-serving-20261018/README.md)执行 `bootstrap-runtime.sh`、应用三个补丁、`build_aclnn.sh`、`build-runtime-bridge.sh`、保存镜像，再运行 `run-gdn-tp2-on-a2.sh`。补丁以本仓库 `patches/qwen38/` 为准，不要重复应用。启动器会停止原 a2 Qwen3.6 服务，重用前检查容器名与设备占用。

下载使用[下载器](../../tools/qwen38_download_checkpoint.py)与[固定 API manifest](../../results/qwen38/tp2-serving-20261018/checkpoint-manifest.json)。在 a2 的仓库根目录执行：

```bash
work=/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018
python3 tools/qwen38_download_checkpoint.py \
 --manifest results/qwen38/tp2-serving-20261018/checkpoint-manifest.json \
 --endpoint https://hf-mirror.com \
 --output /home/ychy/models/Qwen3.8-Flash-Next-W4A16-G128-300i-d32978c \
 --workers 8 --range-chunk-mib 64 --range-workers 8
```

下载器支持断点和有界并行 Range；LFS 核对 SHA256，其他文件核对 Git blob SHA1。全部完成才写 `download-verified.json`。完整权重含真实全量 PLE，不能用算子验证阶段的 sparse fixture。

将[16K 启动器](../../results/qwen38/long-context-20261018/launch-text-tp2-graph-16k-on-a2.sh)、[服务脚本](../../results/qwen38/long-context-20261018/serve-text-tp2-graph-16k.sh)及同目录 `run-long16k-gates.py` 复制到 a2 工作目录；还需 `bench_http_stream.py`、`qwen38_serving_smoke.py`、`check_long_context_needle.py`、`qwen38_graph_request_gate.py`。启动器检查完整下载与两个 rank 的 GDN/掩码门禁，挂载 `/dev/davinci0`、`/dev/davinci1` 及管理设备、驱动库、源码和权重。

| 参数组 | 关键值 |
|---|---|
| TP / EP | `--tensor-parallel-size 2 --enable-expert-parallel --enable-ep-weight-filter` |
| 权重 / 激活 | `cube_310_int4_a8`、`int8_per_group`、G128；非专家、LM head、PLE projection 为 FP16 |
| 容量 | maxlen16384、maxseq1、batch256、grouped chunk256、memory utilization0.95；自动 profile，无显式 KV 预算 |
| 状态 / 通信 | C16、MTP0、Prefix 关闭、mamba cache none；`HCCL_BUFFSIZE=64`、`QWEN38_HCCL_BUFFER_MIB=64` |
| Graph | `VLLM_USE_BREAKABLE_CUDAGRAPH=1`；compilation `mode=0`、`FULL_DECODE_ONLY`、capture `[1]` |
| 请求入口 | `http://127.0.0.1:18038`；模型 `qwen38-w4a8-tp2-text`；仅文本 |

```bash
# 切换前停止当前占用两颗芯片的 Qwen3.8 容器；同名容器已存在时用 podman start。
bash "$work/launch-text-tp2-graph-16k-on-a2.sh"
python3 "$work/run-long16k-gates.py"
# 单独复测，不并发其他请求；输出名须尚不存在。
python3 "$work/bench_http_stream.py" --base-url http://127.0.0.1:18038 \
 --model qwen38-w4a8-tp2-text --prompt-tokens 16256 --output-tokens 128 \
 --repeats 3 --out "$work/long16k-bench-16256-128-new.jsonl"
```

基准为 temperature0、ignore_eos、SSE usage 校验；每组先预热 32 输出 token。Decode=`(output_tokens-1)/(最终usage时刻-首个非空文本时刻)`；输入/TTFT 包含 HTTP、调度和首 token 开销。结果文件拒绝覆盖；门禁重跑也需新结果名或独立目录。4K eager / Graph 脚本分别保留在 [serving](../../results/qwen38/tp2-serving-20261018/README.md) 与 [Graph](../../results/qwen38/decode-graph-20261018/README.md)目录。

## 8. 已知限制与下一步

MTP、C8、Prefix Cache 恢复、并发、多模态及完整质量评估尚未验证。16K 是实测请求长度；32K 当前不可用。三次计时不足以给出正式 P95，短重复文本性能也不能覆盖所有 PLE 访问模式。

按原 Graph 记录的收益与风险顺序推进：8K/16K 检索与生成已完成；下一步先单独对照 **PLE / LM Head 动态 W8A8** 的内存、速度与质量，再评估 **MTP** 的 draft 权重和额外状态，随后做 **C8** 的 QSA 存储/计算协议适配。Prefix Cache 需先解决最少 64 状态槽的容量压力。超过 16K 还要研究物理 KV 分页与长 Prefill 调度。

## 9. 来源与许可

| 来源 | 固定身份 |
|---|---|
| 检查点 | [matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i/tree/d32978ccdde1729caec0034571b03a14779698e9)，revision `d32978ccdde1729caec0034571b03a14779698e9` |
| vLLM fork | [opensensor/vllm](https://github.com/opensensor/vllm/tree/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5)，`3ab5dda29acabea01f6a63d0806bdbbb4a27bde5` |
| Ascend fork | [opensensor/vllm-ascend](https://github.com/opensensor/vllm-ascend/tree/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032)，`ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032`，加本仓库三个补丁 |
| Catlass | `41bf90da655bba3c66d0acd7e00abe33960ecfd6`，与 fork `.gitmodules` 一致 |

许可说明沿用[现有入口页](README.md)：本项目不转载权重，权重与镜像不随仓库分发。

检查点与基础模型 Qwen/Qwen3.8-Flash-Next 均为 **Qwen Community License 1.0**（不是 Apache-2.0）。允许使用、修改和再分发（需保留版权声明）；但若使用者或其关联方经营 *Model as a Service* 或 *AI Work Assistant* 业务并用于商业目的，须另行向 Qwen 取得授权，仅内部使用且不向第三方开放除外。研究与个人复现不受影响；**用于对外提供推理服务的商业场景前须先核对该条款**。
