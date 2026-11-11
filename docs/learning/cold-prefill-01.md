# 第一课：从调度预算到 310P WY 算子

日期：2026-11-11。围绕已有交付制品学习和分析，不新建推理引擎。当前完成源码与制品核对，未新增 NPU 推理或性能测量。

## 本轮固定依据

仓库起点 `c811d1fcb99c4da6a31623c3135aba909e803a33`。镜像 image ID `bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4`；历史 registry digest `sha256:a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5`。本次从 a2 本地 image ID 读取，不声称重新验证远端 registry。容器仅运行 `python3 -S` 读取文件，无网络、无设备映射、不导入 Torch/vLLM；退出后自动删除。

原 `results/wy-mtp-production-20261108/build-manifest.json` 中 43 个文件 SHA 全部匹配。版本记录仍为原验收记录：vLLM 0.23.0+empty、vLLM-Ascend 0.23.0rc2.dev93+g88eac2713、Torch 2.10.0+cpu、torch-npu 2.10.0.post2、镜像 CANN 9.1.0-beta.1；宿主驱动本次 npu-smi 为26.1.1。宿主 stepinfer 实验使用的 CANN 9.1.1 不能混作此镜像版本。

镜像默认：TP2、FP16计算/W8A8量化、262144上下文、seq4、每步2048 token预算、MTP2、主Graph FULL_DECODE_ONLY [3,6,12]、draft eager、prefix off。当前仓库部署manifest显式prefix on；仓库入口脚本本身还默认eager并缺少镜像入口的PR18730 OPP/库路径。它们不是可直接互换的执行配置。本轮仅记录，不重建、不修改这些部署资产。

完整文件SHA与模型摘要见[本轮核对记录](../../results/prefill-learning-20261111/source-audit.json)。镜像原文件只保存在本机忽略目录 `artifacts/prefill-learning-20261111/source/`，不复制进受维护源码。

## 阅读顺序：先看三个文件

以下路径均相对本机上述source目录，行号对应本轮镜像快照。

1. `opt/qwen36/start-wy-mtp.sh`：把环境变量变成vLLM参数，选择设备、OPP与兼容代码。先理解max_model_len、max_num_seqs、max_num_batched_tokens分别限制什么。入口不是调度器，也不执行模型forward。
2. `vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:340` 的 `schedule`：为请求分配本轮token数，消耗总token_budget。源码用已计算token与待计算token的差额统一处理prefill/decode，不是为两者各维护一个完全独立的调度器。
3. `vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py:626`：先理解输入整理、native WY、fwd_h、fwd_o四段。暂时不展开kernel数学。

## 两种chunk不可混淆

调度chunk：本轮所有请求合计受2048 token预算约束。单个4096-token冷文本请求在无命中等简化条件下至少需要两次调度才能处理完prompt；实际还受状态对齐、其他请求、缓存与encoder预算约束。

算子chunk：本镜像 `CHUNK_SIZE=64`，是GDN内部计算与padding的分块。它不意味着调度器每次只送64 token。一个2048-token的单序列计算块对应32个64-token块的规模，但不能把这直接当成kernel launch数。

262144是模型允许的总序列长度上限，4是调度序列上限，2048是单轮token预算；三者描述不同维度。

## 中间桥梁

`vllm/v1/engine/core.py:443` 的step及其batch-queue变体调用scheduler.schedule，再将SchedulerOutput交给model_executor.execute_model。实际是否走batch-queue分支还需运行时证据。

`vllm_ascend/worker/model_runner_v1.py:2783` 的_model_forward准备模型参数并调用self.model。跨进程与TP通信细节留到后续课，不假定这是一个同进程直连栈。

a2权重config声明 `Qwen3_5MoeForConditionalGeneration`，模型有40层：30层linear_attention、10层full_attention。Qwen3.6商品名与代码类名不同，不应凭名称寻找不存在的qwen3_6.py。Qwen3_5DecoderLayer继承Qwen3NextDecoderLayer，后者forward按层类型调用linear_attn或self_attn，然后进入MLP/MoE。

310P GDN实现在 `vllm_ascend/_310p/ops/fla/gdn_310.py:355` 检查num_prefills，调用chunk_gated_delta_rule_310并保存recurrent state。Decode还有独立recurrent调用，不能由WY Prefill优化直接推断Decode加速。

## 算子链与候选观察点

chunk_gated_delta_rule_310中的链：输入规范化/归一化 → varlen与64-token padding → WY生成q/k/w/u/g → fwd_h处理状态 → fwd_o生成输出 →去padding/布局转换。

native WY的入口是 `_compute_kernel_inputs_from_npu_wy:504`。只有device、dtype、shape等条件满足时，才调用 `torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy`，否则仍回退Torch WY；仅启用配置不能证明所有实际输入均命中native。本轮函数docstring还保留“WY用Torch准备”的旧描述，实际应以调用和分支代码为准。

varlen路径有cu_seqlens.cpu/tolist，状态和输出存在transpose/contiguous，且仍有其他矩阵计算、full attention、MoE和通信。这些只是后续profile观察点，不是已证明的瓶颈，不立即改代码。

## 下一次设备窗口

先在a2持有现有 `/tmp/a310-npu.lock`，再次核对rootless/rootful容器和NPU空闲，固定同image ID并使用镜像内入口，独立实验端口。保持主Graph/MTP等配置，不叠加升级或kernel改动。

先做4K冷token-ID输入、单请求、output=1的未采样重复测量与独立profile请求；使用唯一cache_salt并核对prefix hits为0。缓存策略与启动配置记录在窗口中。必要时再扩8K/16K/32K，不先扫参数。纯token-ID实验不覆盖视觉encoder，不称为完整PDF/OCR端到端加速。

复用 `tools/profile_http_prefill.py` 前核对镜像profiler配置和start/stop接口；其profile耗时不可与常规吞吐混报。采样窗口含host与kernel证据，kernel时间累加不直接作为墙钟占比。输出算子累计时间、调用次数、布局转换/拷贝/同步及设备空洞，再决定一个具体优化点。

退出后归档ExitCode与配置/SHA，独立检查容器与NPU清场。a1不操作。当前无新性能数据，不宣称发现新加速。
