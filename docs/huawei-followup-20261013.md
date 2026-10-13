# 华为会议后：MindIEQ QMM 与 GDN PR 独立实验

2026-10-13，a2 / Ascend 310P3，CANN 9.1.0，torch 2.10 / torch-npu 2.10，芯片 0。实验不覆盖已发布模型或宿主 SDK。停止旧 `qwen36-n128-quality` 服务后运行测试。

## 固定来源与计算路径

- [MindIEQ ops-nn](https://gitcode.com/mindIEQ/ops-nn/tree/fake_quant_matmul_x2nz_3_r3_biased)，固定提交 `d9c9871aa335f7b5638e9a2679a45bb3e7561d3a`。
- [GDN PR #17640](https://github.com/vllm-project/vllm-ascend/pull/17640)，固定 head `a2d228c7850f1749a4a934cc7514cd6c01fd7948`，base `75dc5fd3d230aba3c49c8ef74d1e6207b59bfb75`。
- Catlass 固定 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。

MindIEQ 分支不等同于现有 `QwenW4A8Int4MatmulV310`：权重以 biased nibble `(W+8)&15` 存储，设备 UB 解包后生成 L1 INT8 NZ，执行 INT8 Cube，并通过附加纠偏计算减去 `8*sum(A)`。这是 W4 存储、片上 INT8 计算路线，不是现有两路 signed INT4 Cube 的激活高低位组合。其接口限定对称 per-channel 权重 scale、per-token 激活 scale、groupSize=0；无法直接假定支持独立 G128 scale、非零 zero-point 或完整 MoE 调度。

PR #17640 改造 GDN 的 WY、FwdH、FwdO，并处理数值稳定性、因果 mask、尾块与流水线数据。Qwen3.6 配置的 key/value head dim=128、key heads=16、value heads=32 符合 WY 契约。PR 自述的模型/TP/批量不等同于本项目 TP1 35B；不引用其倍数作为本项目端到端收益。

ops-nn/CANN 算子文件使用 CANN Open Software License V2；本仓库保留外部源码引用与测试代码，不将其整体复制成 Apache-2.0 代码。

## 已验证：MindIEQ QMM

隔离编译成功。首次仅设置外置 `ASCEND_CUSTOM_OPP_PATH`，调用失败于 `QuantBatchMatmulV3` tiling compile info parse，错误 561103，未进入矩阵计算。按作者 README 注册容器内 OPP vendor、`config.ini load_priority=custom_nn` 后成功。宿主 SDK 未修改。

作者原始测试：

- `(M,K,N)=(1,2048,2048)、(16,2048,2048)、(1024,2048,2048)` 均 mismatch=0，容差 `atol=rtol=0.01`。
- `(16,2048,2048)` 和 `(1024,2048,2048)` ACL Graph 捕获成功、8 次回放逐位一致。固定输入和地址测试，尚不代表动态输入/完整 vLLM Graph 兼容。
- 20 次同步主机测量：M16 direct 107.0 μs / replay 81.1 μs；M1 direct 104.8 μs / replay 78.9 μs。不是纯 Kernel 延迟。

补充测试：gate/up 合并 K2048/N1024，down K512/N2048；M=1/2/4/8/128/512/1024/4096，INT8 激活覆盖 [-128,127]，所有 16 组 `rtol=atol=.002` 通过，最大绝对误差 <.001。权重仍为随机整数，尚未提取真实 checkpoint 专家。

权重纯数据字节减半：gate/up 2 MiB→1 MiB，down 1 MiB→.5 MiB，不含 scale、工作区和模型其余部分。

同整数/scale 的 W8 对照需要先将转置权重转 FRACTAL_NZ，再取转置视图。初次 ND 调用错误 161002，日志确认 storage format 检查失败；不能将这次失败当作 W8 性能结果。下表来自修正后的 NZ 测试，每组 warmup=5、测量=30，按 W4 后 W8 顺序，无 Graph，包含主机分发、launch、同步，排除离线打包。

| 矩阵 | M | W4 μs | W8 μs | W8/W4 |
|---|---:|---:|---:|---:|
| gate/up | 1 | 93.5 | 73.5 | 0.79 |
| gate/up | 128 | 112.5 | 83.5 | 0.74 |
| gate/up | 1024 | 182.6 | 151.5 | 0.83 |
| gate/up | 4096 | 593.8 | 518.2 | 0.87 |
| down | 1 | 78.3 | 74.0 | 0.94 |
| down | 128 | 86.4 | 102.7 | 1.19 |
| down | 1024 | 167.5 | 141.8 | 0.85 |
| down | 4096 | 655.5 | 569.0 | 0.87 |

大多数形状 W4 较慢，单个 down/M128 较快不足以建立可重复加速结论。也未与本项目现有 native W4 GMM 做同条件对照。下一步应比较 Graph 下的双方及真实权重/多专家路由；不要据此替换模型后端。

## 复现构建

a2 工作目录 `/home/ychy/ascend-lab-runtime/mindieq-r3-20261013`，下列 `/work` 为其容器挂载点。镜像来源 `quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`；仅容器补装 `dos2unix pigz`、禁用损坏 Triton 目录。

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

扩展构建使用外部 `matmul/quant_batch_matmul_v4/tools/torch_extension` 的独立副本 `extension-source`。仅修改 CMake：`aarch64-linux`→`x86_64-linux`；`VENDOR_LIB=/work/ops-nn/build`；`VENDOR_INC=/work/ops-nn/matmul/quant_batch_matmul_v4/op_host/op_api`；移除 include 后缀 `/aclnnop`。

```bash
cmake -S /work/extension-source -B /work/extension-build -DCMAKE_BUILD_TYPE=Release
cmake --build /work/extension-build -j4
# 仅容器内注册内嵌 vendor
cp -a /work/opp/vendors/custom_nn "$ASCEND_HOME_PATH/opp/vendors/custom_nn"
printf 'load_priority=custom_nn\n' > "$ASCEND_HOME_PATH/opp/vendors/config.ini"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
# test-source 为作者测试副本，仅替换 library 路径；环境指定 visible device=0
python /work/test-source/test_qmm_v5_ext.py
python /work/test-source/test_qmm_v5_ext_graph.py
```

作者测试仅打印失败而不保证非零退出码，必须检查 `ALL PASSED` 与 mismatch。补充测试使用 assert、失败会非零退出。

```bash
WORK_DIR=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013 \
  bash experimental/run-mindieq-qmm-check.sh
```

该 runner 使用已登记 vendor 的本机派生镜像 `localhost/qwen36-mindieq-embedded:20261013`，只挂载芯片 0。原始日志与全部样本：`results/mindieq-r3-20261013/`。

## GDN 实验进度

固定 PR archive 独立解包到 `/work/pr17640`。首次编译漏设 Catlass，保留失败日志；补齐锁定依赖后三个算子编译、打包成功。WY 独立 Torch 扩展已编译，使用 `pr17640_wy` namespace，不与官方 `_C_ascend` 冲突；绑定提取保留原版权与许可。

```bash
export CPATH=/work/pr17640/csrc/third_party/catlass/include
cd /work/pr17640/csrc
bash build.sh --pkg \
 --ops='chunk_gated_delta_rule_compute_wy;chunk_gated_delta_rule_fwd_h_vllm;chunk_fwd_o_vllm' \
 --soc=ascend310p -j8
python /project/tools/build_pr17640_wy_extension.py \
 --source-root /work/pr17640 --build /work/wy-extension
```

WY 与原 torch WY 的数值、延迟测试已启动，首次 fallback 触发 CANN 动态编译。128 tokens 的两组先通过 CPU 参考：输出 q/k 精确一致，w/u 最大绝对误差 ≤7.63e-6，g 最大绝对误差 ≤9.54e-6；测试容差 w/u/g 为 rtol=.03、atol=.0002。小衰减时 WY 0.314 ms 对照 9.386 ms，大衰减时 WY 0.233 ms 对照 9.355 ms（5 次 warmup、20 次同步调用）。这仅是 WY 子步骤收益。512/1024 tokens 继续运行中。随后仍需验证 FwdH/FwdO 和真实模型端到端。不将编译成功写成推理成功。

OPP 包 SHA256：`070f2e65f0ce9c82da98d7fc94de1a9738f47126dc9b9b33edbd99c9ae9edcd2`。QMM 扩展 SHA256：`6e59fcbd358ad958ba84a53d3e58bd2fa196ec65959688c5b7bca7a10252ff3f`。Graph debug JSON 仅用于验证 task 名称（实际命中 `QuantBatchMatmulV3_NZ_NZ_int8_int4_fp16`），其中 duration 不作为 profiler 性能数据。

WY 测试命令（同一隔离容器，先安装 PR OPP、设置 vendor 环境）：

```bash
python /project/tools/check_pr17640_wy_310.py \
 --source /work/pr17640/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py \
 --library /work/wy-extension/pr17640_wy.so --out /work/pr17640-wy.jsonl
```

CPU 参考直接提取固定 PR 的原始 torch WY 函数，不复写其数学逻辑；相同函数在 NPU 上作为速度对照。数值用 CPU 比对，速度 warmup 后测量，避免把首次 JIT 编译算入延迟。


## 后续完整 GDN 门禁与模型接入

WY 六组（T128/512/1024 × decay .01/1）全部通过。1024 tokens：原生 WY 1.392–1.518 ms，Torch 对照 22.626–22.741 ms。首次 JIT 时间已排除。

完整 GDN 使用归一化 Q/K、非零 FP16 初始 state、decay O(1)、beta .1–.9，符合 Qwen3.6 TP1 的 16/32 heads、dim128。CPU oracle 是固定 PR 的原始 chunked torch 函数，Q/K 扩展至 value heads。T111/128/512/1024 四组均通过：输出/state cosine >.9999998，五次重复逐位一致；输出 max abs ≤7.63e-6，state ≤6.20e-5。

| T | 完整原生 GDN ms | 相同 FwdH/FwdO + Torch WY ms |
|---:|---:|---:|
| 111 | .842 | 9.947 |
| 128 | .755 | 9.792 |
| 512 | 1.745 | 16.941 |
| 1024 | 3.042 | 24.398 |

全部为 warmup5、重复20的同步主机调用中位数，不等于模型吞吐。独立扩展必须使用 `ASCEND_PLATFORM_310P` 宏，保留 PR 的 initial state NZ staging；仅替换 kernel 而漏掉该绑定逻辑会违反其 state 布局契约。

首次完整测试的独立 namespace 能力探测仍查询原算子名，导致 WY fallback，已终止并重跑；上述结果来自明确选择原生 WY 的修正入口，未使用首次结果。

模型接入通过可选 `QWEN36_GDN_ENABLED=1`，独立 `pr17640_gdn` namespace 加载固定 PR 的三算子；`experimental/pr17640_gdn_adapter.py` 替换 310P chunk GDN Python 入口。原始固定源码在仓外挂载，关闭开关仍走现有路径。OPP 顺序为 PR vendor、W4 vendor，需用实际生成确认默认算子未发生不兼容替换。

模型服务在 a2 `18024` 启动，组合保持 TP1/N128 W4/C8/MTP2/FULL_DECODE_ONLY/112K/.95/batch1024/chunk512/prefix off；目录 `/home/ychy/ascend-lab-runtime/pr17640-product-20261013`。模型端到端尚待验收，不将以上子算子倍数算作模型收益。

真实模型首次启动发现 `gdn_310.py` 的 from-import 提前保留旧函数引用；已替换 chunk 模块、调用方及 package 导出的入口，并通过源码路径断言检查。首次启动未作为性能结果。服务重启后再次核查调用目标。

模型服务已就绪，初步生成8/8通过，A/B/A一致且未观察到canary泄漏。8K/128三次独立前缀：Prefill 495.233/494.504/494.094 tok/s，中位494.504，相对历史同配置N128基线391.591增加26.28%；Decode 36.868/33.494/32.886 tok/s，中位33.494，相对历史37.012约下降9.5%。后两次日志MTP平均接受率约83–85%，不能宣称Decode无回退。比较使用历史同日基线，尚未重新交替A/B测量；32K测试已启动。正式质量与长上下文仍待验证。

32K/128首条样本Prefill446.028、Decode29.000 tok/s；其余重复与相同seed310的GSM32质量回归正在执行，不能将首条当最终中位数。已安排质量评测前后保存MTP指标快照。

32K三次完成：Prefill中位446.028 tok/s、Decode中位28.983 tok/s。完整对照见 `results/pr17640-product-20261013/comparison.json`；GSM32已经开始。

GSM固定32题完成：31/32，与原N128基线相同；仍为index205达到512输出上限、最终答案无法解析，其余31题正确。这是小样本回归，不证明全面精度等价。前后MTP指标已保存。实际114560token检索请求已开始，尚待完成。

本次GSM指标差分：draft cycles2549，draft tokens5098，accepted4596；接受率90.15%，平均接受长度1+4596/2549=2.803。原N128质量记录缺少匹配before，不能构造可靠接受率对照。

实际114560token输入检索完成：返回ASCEND732941，输出14tokens，总114574tokens，耗时352.140s；原N128同一检索请求415.306s，端到端耗时减少15.21%。单条检索不证明最大上下文或完整长文质量；112K边界初步门禁通过。

下一轮仅把MAX_BATCHED_TOKENS从1024调到4096，expert chunk仍512，其他配置保持不变；新目录`/home/ychy/ascend-lab-runtime/pr17640-b4096-20261013`、容器`qwen36-pr17640-b4096`，沿用18024端口。目的为观察降低GDN chunk调用次数的收益、workspace/KV容量代价及Decode/MTP变化。

4096调度批量服务启动、Graph和生成门禁通过；KV容量仍186236tokens，Graph .31GiB。8K/128三次中位Prefill503.099tok/s、Decode37.100tok/s；三个输出与原N128对应输出完全一致。较原N128 Prefill+28.48%、Decode+0.24%；较GDN/b1024 Prefill+1.74%，Decode恢复。此结果提示调度chunk边界可能影响递归数值与MTP接受，尚不能证明已消除所有精度风险。32K、GSM32及实际112K正在顺序验证。

4096调度批量32K/128三次完成：Prefill中位450.652tok/s（相对原N128+26.03%），Decode29.174tok/s（+0.44%）；8K和32K各3条输出全部与原N128对应输出相同。完整三配置比较见`results/pr17640-b4096-20261013/comparison.json`。

4096批量GSM32完成：31/32，仍为index205输出截断；实际114560token检索已开始。质量门禁暂未观察到相对固定基线的得分下降，仍不代表完整评估。

4096批量实际114560token检索通过，耗时348.941s，输出正确检索码14tokens、总114574tokens；相比原N128415.306s减少15.98%，比GDN/b1024的352.140s小幅改善。长请求后的8K复测已启动，独立保留文件以检查运行状态复用。

长请求之后8K复测完成：Prefill中位503.226、Decode37.079tok/s；三条输出与4096候选长请求前全部一致，未观察到持续性能下降。本轮收敛为4096/512单流实验基线，启动器`experimental/run-gdn-112k-product.sh`，参数选择与边界见`docs/gdn-112k-candidate.md`。

新GDN/b4096 MTP1/2业务近似对照已启动：中文解释、会议摘要、Python代码、英文技术分析4类固定chat；每类预热1、测量3、temperature0、thinking关闭、max_tokens256。保存全文、usage、TTFT、TPOT、请求阶段计数和MTP接受率差分，不作为任务质量评分。MTP2初测中位Decode分别31.32/39.22/39.30/34.19tok/s；会议文本重复一致，另外三类有重复输出差异，需与MTP1一起分析。MTP1独立服务正在加载，仅修改draft token数及对应capture sizes。

MTP1/2对照已完成，保留MTP2实验默认。四类chat中位Decode：

| 输入 | MTP1 tok/s | MTP2 tok/s | MTP2变化 |
|---|---:|---:|---:|
| 中文解释 | 30.804 | 31.316 | +1.66% |
| 会议摘要 | 32.960 | 39.225 | +19.01% |
| Python代码 | 33.885 | 39.305 | +16.00% |
| 英文分析 | 32.684 | 34.189 | +4.60% |

8K/128相同prompt token hash三条：MTP1 Prefill510.604、Decode30.571；MTP2 Prefill503.099、Decode37.100tok/s，Prefill -1.47%、Decode +21.36%。三条输出全部跨配置一致。MTP2对照来源是同日4096候选记录，chat是先MTP2再MTP1，未交替A/B。

会议摘要两组均147tokens正常结束、全部重复及跨配置输出一致；其余三类均256tokens上限截断，两组均有重复文本差异，跨配置文本也不相同。不能据此宣称全面质量等价，或将差异全归因于MTP深度。完整全文、接受率及指标见`results/pr17640-mtp-20261013/comparison.json`和原始jsonl。

MTP1启动KV池191507tokens、4.30GiB，MTP2为186236tokens、4.29GiB；两组权重27.288GiB、Graph约.31GiB。池容量均不作为单请求上下文上限。MTP1的TTFT在这四类chat中更低约43–65ms，但MTP2总耗时均较低；主要取舍是草稿成本与批量验证收益。尚未完成MTP0、prefix复用及并发对照。已启动恢复MTP2服务，端口18024。

## 新GDN组合的prefix复用

显式ENABLE_PREFIX_CACHING=1；默认仍0。框架自动选择mamba align和逻辑block1152。TP1/W4/C8/MTP2/Decode Graph/112K/.95/4096:512/maxseq1保持。8K冷TTFT15.839s、热三次中位2.646s（-83.29%）、每次命中6912tokens；32256输入冷68.470s、热11.137s（-83.73%）、每次命中27648。两长度四条输出各自完全一致，Decode基本不变。热请求的logical token/s不是实际冷Prefill算力。

共享文档A/B/A通过，B用独立cache_salt建立冷参考，最终答案严格检查；A三条全文一致、B共享与冷参考全文一致。首次reset_prefix_cache接口404，未计入；B输出32上限曾截断，扩大到256正常回答，原始失败保留。

开启prefix后C8分配116块、单请求预算最低113块；scratch224MiB、Graph.31GiB；启动KV池估计181248tokens不作为请求上限。接近112K实际检索及其后状态检查正在执行；尚未改默认。复现工具bench_prefix_reuse.py与check_prefix_state.py、数据results/pr17640-prefix-20261013。

实际114560token检索通过，正确输出ASCEND732941、总114574tokens、耗时348.211s，关闭prefix同类单样本348.941s。之后A/B/A全部最终答案正确；A三条全文一致，B共享/冷参考的思考段出现少量措辞差异，最终答案LinMei、两条均232输出tokens。严格B全文一致检查为false，未隐藏。补测同一B冷参考的热请求两次全文完全匹配；另一独立cache_salt冷请求全文也有变化、最终答案相同。因此目前不能将变化单独归因于prefix，但也不宣称严格等价。保留prefix为可选试验，默认off。服务18024暂保留prefix候选用于后续诊断。

服务日志归档只归一化换行及行末空白，a2原始log保留。该轮没有证明全面长文质量、最长上下文、并发批处理或P95。

## 真实expert三路公平对照

同真实layer0/expert0权重、同设备pack生成的整数A8与scale，native W4/MindIEQ W4/NZ W8，eager与torch.npu.NPUGraph，四组原地更新输入；独立重复两轮，每轮16形状全部数值通过。gate/up M512 Graph约774/138/116us，M1024约1475/210/163us；down M512约471/127/107us。MindIEQ大M相对当前W4的优势两轮可复现；小M当前W4更快，MindIEQ多数仍慢于W8。排除pack/路由/SwiGLU；单矩阵不能外推模型收益。详细数据、捕获入口首次失败及限制见results/w4-three-backends-20261013。下一步先八expert分组/完整MLP，再条件接入Prefill后端。

部署参数第一步集中到configs/tp1-48g-112k.env，入口scripts/run-candidate.sh --check在a2成功，输出制品SHA256。当前算子实验独占结束，无模型服务；未把新wrapper写成已完整部署验收。

## 完整MLP门禁与256专家占用

已完成八expert真实权重完整MLP。原直接ACL绑定串联动态量化出现Down非有限，调用前同步可消除；隔离补丁改OpCommand::RunOpApiV2排队后，无显式同步10组通过。单矩阵精确整数与完整MLP的独立动态量化不能混为同精度协议；后者相对native L2约1.0%–1.1%，仅诊断门禁。

8专家/4096路由当前W4/MindIEQ约11.32/2.72ms；扩展为256槽均分后约21.72/25.79ms，512路由/256槽约11.00/23.44ms。256槽是重复前8真实专家形成的合成占用，不是完整实际路由。结论：逐专家Python调用方案暂不通过模型接入门禁，优先设备端grouped算子；当前默认后端不替换。证据、补丁工具、失败与复现见results/mindieq-mlp-20261013。

统一run-candidate.sh正在恢复TP1/48G/W4/C8/MTP2/GDN/Graph/4096:512/112K服务，prefix off；首次远端workcopy缺少wrapper所需GDN启动脚本，已同步该现有文件并重启，尚待完成生成与8K复测。

统一run-candidate入口服务启动并完成生成sanity5/5；8K/128三次Prefill中位503.226、Decode37.079tok/s，prompt hashes全部与原基线匹配，输出3/3全文匹配。服务保留18024，prefix关闭。该回归不替代完整质量/长文测试，也不是MindIEQ模型集成。

## 2026-10-13：生命周期与投入收敛

MindIEQ记录inputs/output/workspace stream后的两轮完整MLP混合≥128行，相比原W4在两份counts上省约9%/18%，更分散样本慢约2%～3%；全量MindIEQ两份更慢。仍无端到端收益。异步混合的特定输入还可NaN，定位失败行来自原W4小专家分支；隔离原W4生命周期候选单样本通过，但连续提交测试停滞并停止，未采用。继续投入限于混合数值与设备端grouped门禁，默认服务保持原后端。详见[原始结果与判断](../results/mindieq-lifetime-20261013/README.md)。

## 异步门禁推进

原W4仅recordStream、MindIEQ调用线程分配workspace的隔离组合，三真实counts分布120连续提交输出和16shape/4输入Graph均通过；仍未接入真实服务。混合MLP收益约10%/19%，分散样本慢2.7%，不全量替换。详见[证据](../results/mindieq-queue-localize-20261013/README.md)。

## MindIEQ设备端分组

复用固定外部类，一次launch读取动态专家边界，固定64/128/256 tile。修正ND单行加载与尾行写回后，12整数案例/动态counts Graph通过；全部层0真实256专家权重的完整MLP独立两轮节省约49%～55%耗时，96连续输出/轮通过。仍无模型端到端数据，接入前需处理与native不同的W4布局避免双份权重驻留。详见[分组证据](../results/mindieq-grouped-20261013/README.md)。


## MindIEQ native布局可选Prefill服务

在设备端分组原型基础上，通过GM索引调整与UB signed→biased转换直接读取现有native W4权重，不新增完整专家bank。真实256专家完整MLP三counts约13.262/13.159/10.366ms，原W4约19.176/19.044/17.209ms，耗时减少31%～40%。12整数案例/动态counts Graph与96连续MLP输出通过；native布局目前仅测一轮。

可选TP1服务完成加载、C8缓存规划和Decode Graph capture，权重27.288GB与旧候选相同；基本生成5/5，固定GSM32与旧GDN/4096候选均31/32，答案变化0，既有截断1例，全文一致7/32。小样本不证明全面精度等价。

同一bench_server_phases脚本、8K/128、warmup1/repeats3、三条prompt SHA逐一匹配：Prefill中位629.615tok/s，旧候选503.226；Decode中位35.655 vs37.079。该对照使用同日旧候选结果，尚非交替A/B；当前后端默认关闭，32K与近112K正在验证。初测bench_http_stream的不同提示词结果独立保留，不用于严格加速比。详细代码、原始记录与复现见[本轮实验](../results/mindieq-native-layout-20261013/README.md)。

32K三次已完成：同prompt SHA及旧候选对照，Prefill550.368 vs450.652tok/s（+22.1%），Decode29.070 vs29.174（-0.35%），输出全文3/3一致。8K原始recheck文件实际中位503.226/37.079，候选629.615/35.655，即+25.1%/-3.8%；以service-comparison.json自动计算为准，避免混入其他恢复批次。接近112K检索随后通过。

114560token输入检索同SHA通过，14token输出及答案与旧候选相同，耗时296.351 vs348.941秒（降低15.1%）；长请求后sanity5/5。保留18024实验服务、MindIEQ开关开启，仓库默认关闭。下一步优先配对A/B、8K Decode接受率/输出变化归因及更广质量；不宣布最大上下文或W8对比已完成。

2026-10-14开始B0/A1/B2交替对照，新增逐请求MTP计数。B0 8K/32K复测约630.356/550.744 Prefill、35.797/29.032 Decode tok/s，与上一轮接近；8K接受率91.1%/100%/85.1%，32K全100%。B0/A1 64题质量与原后端/第二次候选复测仍进行中，尚无完整配对结论，见results/mindieq-paired-20261014。


2026-10-14进一步证据：B0固定64题62/64，两题截断；原后端A1已启动，8K接受率全100%、502.936/37.038tok/s，而B0接受率91.1%/100%/85.1%、630.356/35.797。尚需A1质量及B2完成。空闲芯片1的激活量化诊断显示：native在40960合成元素上与FP32 round参考量化码一致，FP16 dynamic_quant有2912差异，主要来自微小幅值输入；预放大128后补偿scale将小幅值重建误差约6.7e-5降至2.7e-7。FP32 dynamic_quant被接口拒绝，仅支持FP16。未把固定放大接入服务（存在大输入溢出风险）；下一步可考虑复用native pack算术生成A8 ND，质量与性能都需门禁。全部数据见paired目录，不能把合成边界诊断写成模型精度结论。


2026-10-14配对质量：A1原后端63/64，B0 MindIEQ62/64，新增711因512输出上限截断；未判默认门禁通过。独立A8 FP32算术准备原型已实现/编译，18shape/CPU量化码/8连续提交/Graph输入更新全部通过，真实256专家完整MLP三个counts约14.000/13.935/11.192ms vs原19.338/19.219/17.285，L2误差降至0.00358%～0.00888%。大M量化准备更慢但MLP仍有局部收益；原型服务尚未启动，详见results/mindieq-a8-quant-20261014。B2仍使用旧dynamic_quant，避免污染交替对照。

B/A/B已全部完成：B0/B2 Prefill相对A1为8K+25.3%/+25.0%、32K+22.3%/+22.0%，32K Decode接近；8K Decode-3.35%/-3.74%，接受率两轮同91.85% vs100%。旧dynamic_quant路径因此保持默认关闭。C3新A8量化库服务已开始独立加载，待711/64题及8K/32K/112K门禁。


C3实际服务启用custom_quant=True、Graph0.30GiB、C8 116块、权重27.288GB。第711题同512上限正确答案6，固定64题C3/A1均63/64，答案变化0、全文26/64一致；未宣称全面精度等价。8K三次Prefill607.019 vs502.936（+20.7%）、Decode34.734 vs37.038（-6.2%），接受率92.96% vs100%，仍有内容相关差异。新A8修复合成小幅值量化及这组新增截断，但未消除模型路径的全部数值/Decode差异；32K及112K门禁运行中。启动器新增显式开关/缺失库preflight错误检查，无NPU初始化。

C3 32K完整三次Prefill533.057 vs450.355（+18.4%）、Decode28.953 vs29.106（-0.52%），MTP100%、输出3/3一致；近112K请求仍执行。当前阶段不满足8K无Decode退化，也未有C3交替重复或更广质量，因此不自动改变默认。

C3近114560输入检索已完成，同SHA正确、303.264秒 vs旧原后端348.941秒（降低13.1%）；长请求后sanity5/5。完整C3分析在results/mindieq-a8-service-20261014/comparison.json。当前继续业务chat补测，基线引用此前同GDN/MTP2/4096候选，必须注明历史对照限制，不宣称随机A/B或全面业务质量。


C3业务补测已完成：962token会议TTFT2.169→1.921秒（-11.4%）、Decode39.225→39.071tok/s；30～45token短输入TTFT增加70～80ms，Decode约-1.7%～+0.7%。这是历史同配置对照，输出有差异，三种场景触及256上限，非业务质量评分。当前建议保留可选长输入Prefill优化，后续验证输入规模分流；不改变默认。完整收束报告、原始输出和分析脚本见[本轮服务证据](../results/mindieq-a8-service-20261014/README.md)。


D4输入规模分流（阈值128）已重新启动并完成业务/8K复测：短输入TTFT恢复至0.393/0.337/0.386秒，会议1.921秒收益保留；8K607.457/34.821tok/s、MTP92.963%重复C3结果。完整64题仍执行中，默认17/off未改，详见[分流实验](../results/mindieq-input-dispatch-20261014/README.md)。


D4完整固定64题完成，A1/D4均63/64、答案变化0、全文39/64一致。芯片1真实8专家逐投影定位：新旧A8码/scale一致，MindIEQ对完整K整数CPU参考10案例逐位一致；native较大M少量FP16差异被CPU G128逐组scale累加参考精确复现。小M native存在channel accumulation分支，因此不能统一用groupwise参考。证据见[逐投影定位](../results/mindieq-projection-parity-20261014/README.md)；尚未证明整模型全部接受率差异来自此处，不将合成投影精度当全面模型质量结论。


Gate/Up token级A8共享（E5）完整性能阶段：8K624.768、32K546.618tok/s，较前轮额外Prefill+2.85%/+2.54%，Decode基本持平，两长度各3/3文本一致；64题仍在运行。a1 W8A8 TP2同输入冷Prefill正式参考为1298.568/1094.336tok/s；每请求独立salt+缓存命中计数为0，初轮未salt的32K有前缀复用可能而不采用。a1与a2存在TP/软件/驱动/KV/MTP差异，不是同机量化验收，详见results/a1-w8-reference-20261014。


E5完整64题已结束，D4/E5均63/64、答案变化0、全文37/64一致。会议TTFT1.921→1.871秒、输出3/3全文一致；原始服务门禁已收齐。保留共享量化为可选优化，默认off，近112K组合及更广质量待补。


E5共享Gate近112K检索追加通过，同SHA114560输入/14输出正确，297.832秒 vsC3 303.264秒/历史native348.941秒；请求后sanity5/5。114560+128持续生成三次已启动，暂不发布未完成速度。

## 2026-10-14 当前服务复测

对正在运行的 18024 候选服务做了新的独占 8K/128 复测（TP1/48G/W4A8/C8/MTP2/Graph、GDN PR、MindIEQ native-layout grouped、prefix off、4096/512）。三次 Prefill 为 625.267/625.386/625.149 tok/s，中位 **625.267 tok/s**；Decode 为 35.811/35.867/35.918 tok/s，中位 **35.867 tok/s**。三次 MTP 计数均为 44 drafts、88 drafted、84 accepted，接受率 95.45%，平均接受长度 2.909。原始记录见 `results/mindieq-current-20261014/current-8k-20261014.jsonl`。该复测用于确认当前候选没有回归，不改变历史配对比较的统计口径。

同一 18024 当前候选服务继续完成 32256/128 复测：Prefill 546.851/546.397/545.844 tok/s，中位 **546.397**；Decode 29.185/29.153/29.136 tok/s，中位 **29.153**。三次 MTP 均 43 drafts、86 drafted、86 accepted，接受率 100%，平均接受长度3。相对同一服务本次 8K 复测，长输入 Prefill 约下降12.6%、Decode约下降18.7%。

当前 18024 候选服务完成 114560 token 近满长单次阶段复测：Prefill **378.373 tok/s**（302.770s），Decode **17.339 tok/s**（0.750s），MTP 5 drafts/10 drafted/10 accepted，接受率100%、平均接受长度3，计数器和 usage 校验通过。该次未携带专门 needle，只用于容量和阶段性能；历史近满长检索正确性记录仍单独保留。
