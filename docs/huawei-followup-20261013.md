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
