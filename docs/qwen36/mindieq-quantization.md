# MindIEQ 长 Prefill 量化方案：从 QMM 验证到输入分流候选

本文整理 MindIEQ 量化路径的完整实验演进，面向推理框架量化与算子优化工程师。数据仅来自以下三份材料；历史阶段的“默认关闭”和后续 `long-prefill` 默认实验档分别保留其语境，不能将后者理解为正式产品验收。

- [MindIEQ QMM 实验原始记录](mindieq-gdn-experiments-20261013.md)：微算子、完整 MLP、设备端分组、激活准备与服务迭代的主要来源。
- [MindIEQ grouped Prefill 评估与投入决策](mindieq-evaluation-20261014.md)：用于交叉核对，并补入后续交替复测、长上下文与质量门禁。
- [112K 候选参数选择](gdn-112k-candidate.md)：仅采用与 MindIEQ 长 Prefill 分流定位和参数选择有关的内容。

版本代号只用于定位证据：下文以“原生 W4 基线”“旧动态量化准备”“FP32 算术 A8 准备”“输入规模分流”“Gate/Up 量化共享”描述方案，在数据处保留 A1、B0/B2、C3、D4、E5。B0 与 B2 是同一旧动态量化路径在 B/A/B 中的前后两轮候选，不应被写成两种新算法。

## 1. 问题：长 Prefill 需要什么样的量化路径

项目已有原生 W4 路径，但后续真实专家对照显示，大 M 矩阵存在优化空间；小 M 则往往仍是原生 W4 更快。因此目标是加速长输入 Prefill，而不是全量替换 W4 后端。最终实现只替换 TP1 routed-expert 的长 Prefill grouped 路径，含 Gate/Up 与 Down 两个投影（见 `experimental/mindieq_prefill_adapter.py` 的 `forward()`）；Decode 和 C8 KV 路径不变，短输入回退原 W4。

MindIEQ 是外部 CANN 算子库，其计算路线与本项目 `QwenW4A8Int4MatmulV310` 不同：

- 权重以 biased nibble `(W+8)&15` 存储，在设备 UB 解包后生成 L1 INT8 NZ，执行 INT8 Cube，再通过附加纠偏减去 `8*sum(A)`。
- 这是 W4 存储、片上 INT8 计算，不是现有两路 signed INT4 Cube 的激活高低位组合。
- 外部接口限定对称 per-channel 权重 scale、per-token 激活 scale、`groupSize=0`；不能直接假定支持独立 G128 scale、非零 zero-point 或完整 MoE 调度。

实验起点为 2026-10-13 的 a2 / Ascend 310P3 / CANN 9.1.0 / torch 2.10 / torch-npu 2.10，使用芯片 0，停止旧 `qwen36-n128-quality` 服务后隔离测试，不覆盖已发布模型或宿主 SDK。外部 [MindIEQ ops-nn](https://gitcode.com/mindIEQ/ops-nn/tree/fake_quant_matmul_x2nz_3_r3_biased) 固定提交为 `d9c9871aa335f7b5638e9a2679a45bb3e7561d3a`，Catlass 固定为 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。

作为背景依赖，项目集成并测试了 [GDN PR #17640](https://github.com/vllm-project/vllm-ascend/pull/17640)（head `a2d228c7850f1749a4a934cc7514cd6c01fd7948`，base `75dc5fd3d230aba3c49c8ef74d1e6207b59bfb75`）；本文不展开该依赖的实现与实验。

ops-nn/CANN 算子文件使用 CANN Open Software License V2；仓库保留外部源码引用与测试代码，不将其整体复制成 Apache-2.0 代码。

## 2. 从隔离 QMM 到真实专家完整 MLP

### 2.1 先确认能计算、能回放，再讨论速度

QMM 隔离编译成功，但首次只设置外置 `ASCEND_CUSTOM_OPP_PATH` 时，调用在 `QuantBatchMatmulV3` tiling compile info parse 阶段失败，错误 561103，尚未进入矩阵计算。按作者 README 注册容器内 OPP vendor，并设置 `config.ini load_priority=custom_nn` 后成功；宿主 SDK 未修改。

作者原始测试覆盖：

| 验证项 | 原始结果 | 适用边界 |
|---|---|---|
| `(M,K,N)=(1,2048,2048)、(16,2048,2048)、(1024,2048,2048)` | 均 mismatch=0，`atol=rtol=0.01` | 隔离数值测试 |
| `(16,2048,2048)`、`(1024,2048,2048)` ACL Graph | 捕获成功，8 次回放逐位一致 | 固定输入和地址，不代表动态输入或完整 vLLM Graph 兼容 |
| 20 次同步主机测量，M16 | direct 107.0 μs / replay 81.1 μs | 非纯 Kernel 延迟 |
| 20 次同步主机测量，M1 | direct 104.8 μs / replay 78.9 μs | 非纯 Kernel 延迟 |

补充测试使用 gate/up 合并 K2048/N1024、down K512/N2048，M=1/2/4/8/128/512/1024/4096，INT8 激活覆盖 [-128,127]。全部 16 组通过 `rtol=atol=.002`，最大绝对误差 <.001。此时权重仍是随机整数，尚未提取真实 checkpoint 专家。

权重纯数据字节减半：gate/up 2 MiB→1 MiB，down 1 MiB→.5 MiB；不含 scale、工作区和模型其余部分。

### 2.2 NZ 格式修正后，W4 也并非普遍快于 W8

同整数/scale 的 W8 对照需要先把转置权重转为 FRACTAL_NZ，再取转置视图。初次 ND 调用错误 161002，日志确认是 storage format 检查失败，不能当成 W8 性能结果。

修正后的 NZ 测试每组 warmup=5、测量=30，按 W4 后 W8 顺序，无 Graph；包含主机分发、launch、同步，排除离线打包：

| 矩阵 | M | MindIEQ W4 μs | NZ W8 μs | W8/W4 |
|---|---:|---:|---:|---:|
| gate/up | 1 | 93.5 | 73.5 | 0.79 |
| gate/up | 128 | 112.5 | 83.5 | 0.74 |
| gate/up | 1024 | 182.6 | 151.5 | 0.83 |
| gate/up | 4096 | 593.8 | 518.2 | 0.87 |
| down | 1 | 78.3 | 74.0 | 0.94 |
| down | 128 | 86.4 | 102.7 | 1.19 |
| down | 1024 | 167.5 | 141.8 | 0.85 |
| down | 4096 | 655.5 | 569.0 | 0.87 |

大多数形状 W4 较慢，单个 down/M128 较快不足以建立可重复加速结论。这一轮也未与本项目 native W4 GMM 做同条件对照，不能据此替换模型后端。

### 2.3 真实权重证明大 M 的机会，也暴露逐专家调用的代价

随后采用真实 layer0/expert0 权重、同设备 pack 生成的整数 A8 与 scale，对照 native W4 / MindIEQ W4 / NZ W8，覆盖 eager 与 `torch.npu.NPUGraph`、四组原地更新输入。独立重复两轮，每轮 16 形状全部数值通过。

| 真实权重 Graph 场景 | native W4 / MindIEQ W4 / NZ W8 |
|---|---|
| gate/up M512 | 约 774/138/116 μs |
| gate/up M1024 | 约 1475/210/163 μs |
| down M512 | 约 471/127/107 μs |

MindIEQ 大 M 相对当前 W4 的优势两轮可复现；小 M 当前 W4 更快，MindIEQ 多数仍慢于 W8。测试排除了 pack、路由和 SwiGLU，单矩阵不能外推模型收益。详细数据、捕获入口首次失败与限制见 `results/w4-three-backends-20261013`。

完整 MLP 才是下一道门。八专家真实权重测试中，原直接 ACL 绑定串联动态量化出现 Down 非有限值，调用前同步可消除；隔离补丁改为 `OpCommand::RunOpApiV2` 排队后，无显式同步 10 组通过。单矩阵精确整数协议与完整 MLP 的独立动态量化协议不同，后者相对 native L2 约 1.0%–1.1%，当时仅为诊断门禁。

| 完整 MLP 占用 | 当前 W4 / MindIEQ | 结论边界 |
|---|---|---|
| 8 专家 / 4096 路由 | 约 11.32/2.72 ms | 少专家下有局部收益 |
| 256 槽均分 | 约 21.72/25.79 ms | 扩大专家占用后反而更慢 |
| 512 路由 / 256 槽 | 约 11.00/23.44 ms | 更分散占用未通过接入门禁 |

256 槽是重复前 8 个真实专家形成的合成占用，不是完整实际路由。逐专家 Python 调用方案因此暂不通过模型接入门禁，后续优先设备端 grouped 算子，不替换默认后端。证据、补丁工具、失败与复现见 `results/mindieq-mlp-20261013`。

这一阶段部署参数集中到 `configs/tp1-48g-112k.env`，`scripts/run-candidate.sh --check` 在 a2 成功并输出制品 SHA256；wrapper 尚不能算完整部署验收。恢复服务首次远端 workcopy 缺少 wrapper 所需启动脚本，补齐现有文件后重启。统一入口随后 sanity5/5，8K/128 三次 Prefill 中位 503.226、Decode 37.079 tok/s，prompt hashes 与原基线全部匹配、输出 3/3 全文匹配；TP1/48G/W4/C8/MTP2/Graph/4096:512/112K、prefix off、端口 18024。该回归不是 MindIEQ 模型集成，也不替代完整质量或长文测试。

## 3. 决策一：设备端分组，消除逐专家 Python 调用

### 3.1 先解决异步生命周期，再评估混合方案

记录 inputs/output/workspace stream 后，两轮完整 MLP 的混合方案在 ≥128 行时，相比原 W4 在两份 counts 上省约 9%/18%，更分散样本慢约 2%～3%；全量 MindIEQ 两份更慢，仍无端到端收益。异步混合的特定输入还可出现 NaN，失败行定位到原 W4 小专家分支；隔离原 W4 生命周期候选单样本通过，但连续提交测试停滞并停止，未采用。详见[原始结果与判断](../../results/qwen36/mindieq-lifetime-20261013/README.md)。

后续隔离组合采用原 W4 仅 `recordStream`、MindIEQ 调用线程分配 workspace：三真实 counts 分布 120 连续提交输出、16shape/4 输入 Graph 均通过，仍未接入真实服务。混合 MLP 收益约 10%/19%，分散样本慢 2.7%，不全量替换。详见[异步证据](../../results/qwen36/mindieq-queue-localize-20261013/README.md)。

### 3.2 一次 launch 读取动态专家边界

设备端分组复用固定外部类，一次 launch 读取动态专家边界，固定 64/128/256 tile。修正 ND 单行加载与尾行写回后，12 整数案例、动态 counts Graph 通过；全部层 0 真实 256 专家权重完整 MLP 独立两轮节省约 49%～55% 耗时，96 连续输出/轮通过。当时仍无模型端到端数据，且须处理与 native 不同的 W4 布局，避免双份权重驻留。详见[分组证据](../../results/qwen36/mindieq-grouped-20261013/README.md)。

通过 GM 索引调整与 UB signed→biased 转换，后续版本可直接读取现有 native W4 权重，不新增完整专家 bank。真实 256 专家完整 MLP 三 counts 为约 13.262/13.159/10.366 ms，原 W4 约 19.176/19.044/17.209 ms，耗时减少 31%～40%；12 整数案例、动态 counts Graph、96 连续 MLP 输出通过。native 布局此时仅测一轮。

### 3.3 接入服务后，Prefill 收益与 Decode 差异同时出现

native-layout 可选 TP1 服务完成加载、C8 缓存规划和 Decode Graph capture，权重 27.288GB 与旧候选相同。基本生成5/5；固定 GSM32 与旧候选均31/32，答案变化0，既有截断1例，全文一致7/32，不证明全面精度等价。

同一 `bench_server_phases` 脚本，warmup1/repeats3，三条 prompt SHA 逐一匹配：

| 历史同日对照 | native-layout 候选 | 旧原生 W4 候选 | 原文变化 |
|---|---:|---:|---|
| 8K/128 Prefill | 629.615 tok/s | 503.226 tok/s | +25.1% |
| 8K/128 Decode | 35.655 tok/s | 37.079 tok/s | -3.8% |
| 32K/128 Prefill | 550.368 tok/s | 450.652 tok/s | +22.1% |
| 32K/128 Decode | 29.070 tok/s | 29.174 tok/s | -0.35% |

32K 输出全文3/3一致。8K 以原始 recheck 中位 503.226/37.079 和 `service-comparison.json` 自动计算为准，避免混入其他恢复批次。初测 `bench_http_stream` 的不同提示词记录独立保留，不用于严格加速比。该轮使用同日旧候选结果，并非交替 A/B。详细代码、记录与复现见[本轮实验](../../results/qwen36/mindieq-native-layout-20261013/README.md)。

114560 token 输入检索同 SHA 通过，14 token 输出及答案与旧候选相同，耗时 296.351 vs 348.941 秒（降低15.1%），长请求后 sanity5/5。端口18024保留实验服务、MindIEQ 开关开启，仓库默认关闭；尚不能宣布最大上下文或 W8 对比已完成。下一步是配对 A/B、8K Decode 接受率/输出差异归因和更广质量。

## 4. 决策二：小幅值误差需要 FP32 算术 A8 准备

### 4.1 旧动态量化准备的配对回归没有通过默认门禁

B0/A1/B2 交替对照新增逐请求 MTP 计数。B0 8K/32K Prefill 约 630.356/550.744、Decode 35.797/29.032 tok/s；8K 接受率三次为91.1%/100%/85.1%，32K 全100%。原后端 A1 的8K 接受率全100%，Prefill/Decode 为502.936/37.038 tok/s。原始过程见 `results/mindieq-paired-20261014`。

固定64题中，A1 原后端63/64，B0 MindIEQ62/64、两题截断；新增第711题因512输出上限截断，因此未判默认门禁通过。B2 继续使用旧 `dynamic_quant`，避免污染交替对照。B/A/B 完成后，B0/B2 相对 A1 的8K Prefill +25.3%/+25.0%、32K +22.3%/+22.0%，32K Decode 接近；8K Decode -3.35%/-3.74%，两轮接受率同为91.85% vs100%。旧动态量化路径因此保持默认关闭。

### 4.2 合成诊断找到小幅值问题，但不能直接等同模型归因

空闲芯片1的诊断中，native 在40960个合成元素上与 FP32 round 参考量化码一致，FP16 `dynamic_quant` 有2912个差异，主要来自微小幅值输入。预放大128再补偿 scale，将小幅值重建误差约6.7e-5降至2.7e-7。

FP32 `dynamic_quant` 被接口拒绝，该接口仅支持 FP16。固定放大存在大输入溢出风险，未接入服务。后续独立实现 A8 FP32 算术准备，而不是把合成诊断写成模型精度结论。

该原型实现/编译后，18shape/CPU量化码/8连续提交/Graph输入更新全部通过。真实256专家完整MLP三 counts 约14.000/13.935/11.192 ms，对照原19.338/19.219/17.285 ms，L2误差降至0.00358%～0.00888%。大 M 量化准备更慢，但完整 MLP 仍有局部收益。原型阶段证据见 `results/mindieq-a8-quant-20261014`。

### 4.3 FP32 算术服务修复新增截断，仍保留 Decode 差异

新 A8 准备服务 C3 启用 `custom_quant=True`，Graph0.30GiB、C8 116块、权重27.288GB。第711题在相同512上限下正确回答6；固定64题 C3/A1 均63/64，答案变化0，全文26/64一致。

8K 三次 Prefill607.019 vs502.936 tok/s（+20.7%），Decode34.734 vs37.038（-6.2%），接受率92.96% vs100%；32K 三次 Prefill533.057 vs450.355（+18.4%），Decode28.953 vs29.106（-0.52%），MTP100%、输出3/3一致。新 A8 修复合成小幅值量化及这组新增截断，但未消除全部模型数值/Decode差异，不满足8K无Decode退化，也尚无 C3 交替重复或更广质量，因此不自动改变默认。

启动器增加显式开关、缺失库 preflight 错误检查，不初始化 NPU。近114560输入检索同SHA正确，303.264秒 vs旧原后端348.941秒（降低13.1%）；长请求后sanity5/5。完整分析见 `results/mindieq-a8-service-20261014/comparison.json`，原始输出与分析脚本见[本轮服务证据](../../results/qwen36/mindieq-a8-service-20261014/README.md)。基线引用此前同配置候选，必须保留历史对照限制，不能称随机 A/B 或全面业务质量。

## 5. 决策三：用输入规模分流保留长输入收益

C3 业务补测中，962token会议 TTFT2.169→1.921秒（-11.4%），Decode39.225→39.071 tok/s；但30～45token短输入 TTFT 增加70～80ms，Decode约-1.7%～+0.7%。这是历史同配置对照，输出有差异，三种场景触及256上限，不是业务质量评分。

这一结果推动阈值128的输入规模分流 D4：长输入走 MindIEQ，短输入回退原 W4。复测后短输入 TTFT 恢复到0.393/0.337/0.386秒，会议1.921秒收益保留；8K Prefill/Decode607.457/34.821 tok/s，MTP92.963%，重复 C3 表现。该阶段原文记录“默认17/off未改”。证据见[分流实验](../../results/qwen36/mindieq-input-dispatch-20261014/README.md)。

D4 固定64题与 A1 均63/64、答案变化0、全文39/64一致。芯片1真实8专家逐投影定位表明：新旧A8码/scale一致，MindIEQ 对完整K整数CPU参考10案例逐位一致；native 较大M的少量FP16差异被CPU G128逐组scale累加参考精确复现。小M native 存在 channel accumulation 分支，不能统一使用 groupwise 参考。尚未证明全部模型接受率差异都来自此处。详见[逐投影定位](../../results/qwen36/mindieq-projection-parity-20261014/README.md)。

`MIN_TOKENS=128` 是这轮输入分流实测采用的阈值，不是全局最优搜索结论。最终定位为长 Prefill 候选分流，保留原 W4 回退。

## 6. 决策四：Gate/Up 共享 token 级 A8 准备

在分流方案上，E5 复用 Gate/Up 的 A8 激活与 scale，保留原 W4 权重存储。实现位于 `experimental/mindieq_grouped_kernel/` 和 `experimental/mindieq_prefill_adapter.py`。

17/32/128/512 token 设备微测中，A8 codes/scale 和 Gate output 均 bitwise identical；512 token 的 Gate+准备中位耗时由4.406ms降为3.172ms。这为复用准备开销提供局部证据。

| 冷前缀 | 前一 grouped 版本 Prefill | E5 Prefill | 增量 | Decode 增量 | 文本一致 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 607.46 tok/s | 624.77 tok/s | +2.85% | -0.40% | 3/3 |
| 32256 | 533.06 tok/s | 546.62 tok/s | +2.54% | +0.03% | 3/3 |

主要记录保留的 E5 原始精度为8K624.768、32K546.618 tok/s。测试为独立服务、prefix off、warmup1/repeat3，并非交替 A/B，增益是方向性证据，不是正式置信区间。原始数据见 `results/qwen36/mindieq-shared-gate-20261014/service-comparison.json`、`shared-gate-v1-20261014.txt`，参数材料另保留 `results/mindieq-shared-gate-20261014` 引用。

D4/E5 固定64题均63/64，`answer_changes=[]`，全文37/64一致；会议TTFT1.921→1.871秒，输出3/3全文一致。原始服务门禁收齐时，共享量化仍是可选优化、默认off，更广质量待补。

E5 近112K检索追加通过：同SHA114560输入/14输出正确，297.832秒，对照C3 303.264秒/历史native348.941秒；请求后sanity5/5。此时114560+128持续生成三次仅已启动，原记录未发布其未完成速度；后续完成的独立复测见下文，不能倒填为这一轮结果。

a1 W8A8 TP2 同输入冷Prefill参考为1298.568/1094.336 tok/s，每请求独立salt、缓存命中计数为0；初轮未salt的32K有前缀复用可能，不采用。a1与a2存在TP/软件/驱动/KV/MTP差异，不是同机量化验收。详见 `results/a1-w8-reference-20261014`。

## 7. 最终效果：按比较协议分别阅读

### 7.1 方案演进中的配对质量和性能

下表保留版本以追溯配置；各行沿用其自身基线与测试协议，不能横向拼成统一加速比。Prefill/Decode 单位均为 tok/s，百分比直接沿用原文，不重新计算。

| 方案与证据阶段 | 8K Prefill / Decode | 32K Prefill / Decode | 配对质量及限制 |
|---|---|---|---|
| 原生 W4 基线 A1 | 502.936 / 37.038；8K接受率全100% | C3对照引用450.355 / 29.106 | 固定64题63/64 |
| native-layout初次服务、旧动态量化 | 629.615 / 35.655，历史对照+25.1%/-3.8% | 550.368 / 29.070，历史对照+22.1%/-0.35% | GSM32双方31/32，答案变化0、全文7/32；32K全文3/3 |
| 旧动态量化 B0/B2，B/A/B | B0约630.356 / 35.797；相对A1两轮Prefill+25.3%/+25.0%，Decode-3.35%/-3.74% | B0约550.744 / 29.032；相对A1两轮Prefill+22.3%/+22.0%，Decode接近 | B0 62/64 vs A1 63/64，新增711截断；8K两轮接受率91.85% vs100%，默认门禁未通过 |
| FP32算术A8准备 C3 | 607.019 / 34.734，+20.7%/-6.2% | 533.057 / 28.953，+18.4%/-0.52% | C3/A1均63/64，答案变化0、全文26/64；32K MTP100%、全文3/3 |
| 输入规模分流 D4，阈值128 | 607.457 / 34.821，MTP92.963% | 前一版本比较表引用533.06 Prefill | D4/A1均63/64，答案变化0、全文39/64；短输入TTFT恢复 |
| Gate/Up量化共享 E5 | Prefill624.768，相对前轮+2.85%，Decode增量-0.40% | Prefill546.618，相对前轮+2.54%，Decode增量+0.03% | D4/E5均63/64，答案变化0、全文37/64；两个长度各3/3文本一致 |

64题与32题门禁均为固定小子集；同分与答案不变不等于全文一致或全面精度等价。尤其不能用后续质量改善抹去旧动态量化路径的失败。

### 7.2 相对原始 W4 的后续交替与长上下文结果

早期原始 W4 到 grouped 路径的历史提升约22%～25%；E5 相对前一 grouped 版本的约2.5%～2.9%仅为增量优化，不能代替原始 W4 对照。后续交替复测支持将 MindIEQ 保留为产品候选，但仍有业务质量与尾延迟门禁。

| 协议与输入/输出 | MindIEQ Prefill | 原始 W4 Prefill | 原文提升 | Decode 与输出证据 |
|---|---:|---:|---:|---|
| 8K，轮换各5次 | 620.27 | 503.09 | +23.29% | Decode变化-0.06% |
| 32K，轮换各5次 | 541.99 | 449.07 | +20.69% | Decode变化-0.48% |
| 64K（64512/128），单流长请求 | 463.77 | 394.39 | +17.59% | 两边Decode 22.638/22.583，差异+0.24%，均完整返回 |
| 近112K（114560/14），单流长请求 | 377.18 | 328.94 | +14.66% | 两边Decode 17.333/17.349，差异−0.09%，均完整返回 |
| 64K，非流式completion、严格同token、5轮交替 | 483.35 | 395.66 | +22.16% | Decode22.60 vs22.40，输出哈希一致 |
| 112K，离线token-ID、近上限3轮交替 | 378.34 | 329.53 | +14.81% | Decode16.96 vs16.83，输出哈希一致；仅32输出token |
| 114560输入/128输出，同token、2轮交替 | 378.69 | 329.60 | +14.89% | Decode16.87 vs16.77（+0.63%），四次输出哈希一致，服务恢复空闲 |

前两行使用同一批token，在a2两颗同型号310P的18024（E5）和18028（原始W4）间轮换，各prompt长度五次；不能写成同芯片交替。后续64K经验尾部值不等同线上并发P95。112K离线token-ID复测消除了客户端准备阶段歧义，早期“tokenizer阻塞”判断证据不足。

长配对原始记录与说明：`results/qwen36/mindieq-long-paired-20261014/paired-64k-ns-v4.jsonl`、`results/qwen36/mindieq-long-paired-20261014/paired-112k-fixed-o128-v1.jsonl`、`results/qwen36/mindieq-long-paired-20261014/README.md`。32输出token结果不能替代持续Decode，128输出token的两轮结果仍不能替代业务质量、流式尾延迟和多请求门禁。

### 7.3 当前服务复测与内容门禁，单独保留口径

2026-10-14 对18024当前候选做独占复测：TP1/48G/W4A8/C8/MTP2/Graph、MindIEQ native-layout grouped、prefix off、4096/512，保留已集成背景依赖。该轮用于确认当前候选无回归，不改变历史配对比较口径。

| 当前候选复测 | 三次Prefill / 中位（tok/s） | 三次Decode / 中位（tok/s） | MTP |
|---|---|---|---|
| 8K/128 | 625.267/625.386/625.149，中位625.267 | 35.811/35.867/35.918，中位35.867 | 每次44 drafts/88 drafted/84 accepted，95.45%，平均接受长度2.909 |
| 32256/128 | 546.851/546.397/545.844，中位546.397 | 29.185/29.153/29.136，中位29.153 | 每次43 drafts/86 drafted/86 accepted，100%，平均接受长度3 |

相对同一服务本轮8K，32K Prefill约下降12.6%、Decode约下降18.7%。8K原始记录见 `results/qwen36/mindieq-current-20261014/current-8k-20261014.jsonl`。

114560 token近满长单次阶段复测 Prefill378.373 tok/s（302.770s）、Decode17.339 tok/s（0.750s），MTP5 drafts/10 drafted/10 accepted，100%、平均接受长度3，计数器和usage校验通过。该次无专门needle，仅用于容量与阶段性能。

随后当前服务在114560 token输入中部插入 `ASCEND732941`，耗时297.733s，正确返回验证码，`retrieved=true`，usage114560/14、正常stop。记录见 `results/qwen36/mindieq-current-20261014/current-114560-needle-20261014.jsonl`。这支持近满长度基本检索链路可用，不替代全面长上下文质量集合。

后续评估材料补充了以下配对内容门禁，不能与上述单服务记录混算：

- 112K needle：同114560 token文档、needle在73%位置、输出上限14，两边均返回 `ASCEND732941`，usage与stop原因一致；MindIEQ298.08s、原始W4 348.15s。单一canary不替代完整质量集或多位置测试。
- 32K多位置needle：25%/50%/75%/90%四个位置，MindIEQ与原始W4共8次请求全部正确，usage与stop原因一致；仍不是完整任务质量集。
- 112K合成会议决议更新canary：两边均正确合并三段更新，输出相同54token，草稿接受率均100%；Prefill344.68/311.06 tok/s（约+10.8%），Decode16.49/16.35 tok/s。该输入收益低于重复合成吞吐样本，仍须真实业务语料验收；完成后两端空闲、健康检查200。输入、计数与命令见 `results/qwen36/mindieq-long-paired-20261014/README.md`。

## 8. 候选参数、回退与已知限制

最终保留 MindIEQ 为 `MIN_TOKENS=128` 的长 Prefill 分流，短输入回退原W4。最可靠回退基线仍是原W4A8+C8+已集成背景依赖+MTP2+Decode Graph。后续评估材料记录长上下文业务将 `long-prefill` 设为唯一默认实验部署档，自动选择E5与阈值128，内部原W4/MTP1对照保留；该默认档变更没有重启正在提供对照的两端服务，也未改变已测kernel制品。基础W4组合继续承担回归用途，默认实验档不等于正式产品验收。

与分流有关的已测部署参数为TP1、单颗48G、N128 W4A8（MTP保留浮点权重）、实验动态C8、MTP2、`FULL_DECODE_ONLY`/mode0/capture1/2/3、max_model_len114688（112 Ki tokens）、gpu_memory_utilization .95、调度批量4096、专家chunk512、prefix off、max_num_seqs1。这里的C8不是官方310P C8后端，.95不表示.90一定不可用；参数是已测工程选择，不声称全局最优。参数来源与候选入口见[参数选择](gdn-112k-candidate.md)、`experimental/run-gdn-112k-product.sh`、`scripts/run-candidate.sh`。

已知限制和未完成门禁集中如下：

- 单矩阵排除pack/路由/SwiGLU，不能外推模型收益；权重字节减半也不是模型总内存减半。随机权重、真实单专家、合成256槽、全部层0真实专家分别属于不同证据层次。
- 历史同日同配置对照不是严格A/B；E5相对前一grouped版本是独立服务warmup1/repeat3，只有方向性增益。后续轮换测试使用两颗同型号芯片，也不应改写为同芯片实验。
- 旧动态量化路径存在新增截断与8K Decode回退；FP32算术准备修复了特定小幅值问题，但早期8K Decode仍退化。逐投影参考尚未解释整模型全部接受率差异；不能以局部bitwise parity替代全面模型质量。
- GSM32、固定64题、有限业务文本、单一或多位置needle、合成会议canary均不足以覆盖长上下文、多任务、异常路由和完整业务质量。输出上限截断与全文差异必须保留；同分不代表全文等价。
- 仅验证TP1、FP16激活、固定checkpoint形状、单流；尚不支持TP2/C8多请求，不能外推并发吞吐、48G之外32G硬件或官方main迁移。跨a1/a2的W8A8 TP2数据不是同机量化验收。
- 依赖外部MindIEQ制品与实验性monkeypatch，需补齐正式交付的制品hash、许可证和回退说明。外部接口的scale与zero-point契约也不能越界假定。
- P95、完整业务质量集、TP2/C8与并发门禁尚未完成；经验尾部值不是线上并发P95，短输出或有限轮次长输出不等于持续Decode稳定性，canary与vLLM缓存估计不能证明最大上下文。
- 正式产品化或单独提交上游前，原评估要求同芯片或可控隔离交替A/B，8K/32K/64K/112K各至少5次、Prefill P50稳定提升≥10%；Decode tok/s、TTFT、MTP接受率无统计显著退化；公开质量集与长上下文needle通过且无输出漂移。现有各长度的次数和协议不同，不能写成全部完成。

下一优先级沿用评估材料：同协议真实会议摘要/待办提取质量、prefix增量复用、长输出稳定性，之后再做并发与尾延迟；达到门槛前优先kernel/dispatch profile与交替A/B，不盲目扩展完整模型集成范围。

## 附：QMM 隔离构建与复现入口

为保留微算子阶段的复现链路，以下记录沿用原始环境。a2工作目录 `/home/ychy/ascend-lab-runtime/mindieq-r3-20261013`，`/work`是容器挂载点；镜像 `quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，仅容器补装 `dos2unix pigz`、禁用损坏Triton目录。

```bash
# 外部目录；固定 revision
 git clone -b fake_quant_matmul_x2nz_3_r3_biased https://gitcode.com/mindIEQ/ops-nn.git ops-nn
 git -C ops-nn checkout d9c9871aa335f7b5638e9a2679a45bb3e7561d3a
# 以下在隔离容器运行
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/ops-nn
bash build.sh --pkg --ops=quant_batch_matmul_v4 --soc=ascend310p -j8
bash build_out/cann-ops-nn-custom_linux-x86_64.run --install-path=/work/opp
```

扩展构建使用外部 `matmul/quant_batch_matmul_v4/tools/torch_extension` 的独立副本 `extension-source`；仅修改CMake：`aarch64-linux`→`x86_64-linux`，`VENDOR_LIB=/work/ops-nn/build`，`VENDOR_INC=/work/ops-nn/matmul/quant_batch_matmul_v4/op_host/op_api`，移除include后缀 `/aclnnop`。

```bash
cmake -S /work/extension-source -B /work/extension-build -DCMAKE_BUILD_TYPE=Release
cmake --build /work/extension-build -j4
# 仅容器内注册内嵌 vendor
cp -a /work/opp/vendors/custom_nn "$ASCEND_HOME_PATH/opp/vendors/custom_nn"
printf 'load_priority=custom_nn\n' > "$ASCEND_HOME_PATH/opp/vendors/config.ini"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
# test-source 为作者测试副本，仅替换 library 路径；visible device=0
python /work/test-source/test_qmm_v5_ext.py
python /work/test-source/test_qmm_v5_ext_graph.py
```

作者测试只打印失败而不保证非零退出码，须检查 `ALL PASSED` 与mismatch；补充测试使用assert、失败非零退出。

```bash
WORK_DIR=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013 \
  bash experimental/run-mindieq-qmm-check.sh
```

runner使用已登记vendor的本机派生镜像 `localhost/qwen36-mindieq-embedded:20261013`，只挂载芯片0。原始日志与全部样本见 `results/qwen36/mindieq-r3-20261013`。

原记录保留的OPP包SHA256为 `070f2e65f0ce9c82da98d7fc94de1a9738f47126dc9b9b33edbd99c9ae9edcd2`，QMM扩展SHA256为 `6e59fcbd358ad958ba84a53d3e58bd2fa196ec65959688c5b7bca7a10252ff3f`。Graph debug JSON仅验证实际命中task `QuantBatchMatmulV3_NZ_NZ_int8_int4_fp16`，其duration不作为profiler性能数据。
