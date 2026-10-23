# 真实 Prefill 路由与 MindIEQ 完整 MLP 门禁

2026-10-13，a2 / Ascend310P3 / TP1。默认服务仍使用原有 W4 grouped 算子；MindIEQ 尚未接入 vLLM。

## 路由证据

W4A8 + C8 + MTP2 + Decode Graph + GDN，max_model_len=114688、max_num_batched_tokens=4096、expert chunk=512、prefix 关闭。通过独立 `experimental/route_occupancy_trace.py` 包装 `_forward_grouped`，只在 `/tmp/route-occupancy.jsonl.enabled` 存在时统计。启动探测与捕获期间不采集；默认不导入该模块。诊断含 device-to-host 同步，不能用于性能比较。没有修改主模型专家计算。

三类重复文本构成的8192-token输入，每次生成8 tokens，覆盖40个MoE模块、640个chunk；合计1920个chunk。模块编号是首次采集顺序，不是经模型命名核实的layer id。非真实客户语料，不能代表全部路由分布；不采集激活或客户数据。

| 输入 | 活跃专家中位数 | 每块最大专家行数中位数 | 路由落在≥128行专家的比例 |
|---|---:|---:|---:|
| 技术说明 | 108 | 281.5 | 32.56% |
| 中文会议 | 173.5 | 233 | 18.03% |
| Python代码 | 174 | 224.5 | 14.20% |
| 合计 | 159 | 245 | 21.60% |

合计≥256行比例6.70%。不等于“只有21.6%的计算可以优化”，也不推算Amdahl端到端加速。每个chunk512 token × top-k8=4096 routes；counts求和、专家数量和timing_valid均校验。

## 完整 MLP 实测

在合计采样中按“最大专家行数”排序，取p05/p50/p95三个真实counts向量。这里p95是路由分布排序位置，**不是性能P95**。输入为合成FP16激活；先重复层0的8专家做探索，再用层0全部256个真实专家权重复测。计入激活量化、Gate/Up、FP32 SwiGLU→FP16、Down、逐专家Python提交与混合拆分/合并；不含真实router、共享专家、Attention、token route combine。权重事先打包，256专家全部驻留。

全部真实专家、warmup5次、重复20次、交错后端顺序，中位数如下（ms）：

| 分布 | 当前W4 grouped | 全部MindIEQ | W8计算对照 | 混合≥128行 | 混合≥256行 |
|---|---:|---:|---:|---:|---:|
| p05 | 19.198 | 18.440 | 15.701 | 19.615 | 19.628 |
| p50 | 19.018 | 16.657 | 14.069 | 17.263 | 19.522 |
| p95 | 数值比较参考有限值 | **MindIEQ NaN，门禁失败** | 未完成该组比较 | 未完成该组比较 | 未完成该组比较 |

p50混合≥128行约减少9.2%耗时；全部MindIEQ约减少12.4%。p05混合增加约2.2%耗时。两组不能证明稳定模型收益。W8对照是同一W4整数权重展开到INT8的算子对照，**不是原W8A8模型质量或端到端速度**。

MindIEQ与当前W4使用不同per-token激活量化器，通过案例相对L2误差约1.06%，不是共享整数输入下的kernel计算误差。质量与Graph门禁未完成。混合诊断使用预知的CPU counts，不允许直接照搬到正式decode路径。

## 数值阻塞与诊断

重复8权重与全部256真实权重两次完整运行，均在p95首次MindIEQ输出比较出现NaN；前两个分布正常。原始错误保存在 `failure-real-weights.txt`；JSONL没有PASS summary，不能把前两个PASS当整组成功。

单独p95诊断中，逐投影有限值同步、调用前同步生产者、保留中间张量直到完整MLP末尾同步，均出现过PASS；均不作为有效测速。单独p95默认随机输入是seed310的第一份；原三分布p95为第三份，因此增加 `--activation-draw 2` 保持CPU随机序列一致，再对照保留张量与普通异步调用。同输入（activation-draw=2）结果：普通异步调用再次NaN，保留中间张量到完整MLP末尾同步则PASS（relative L2约1.045%）。见 `failure-plain-matched.txt` 和 `mindieq-full-mlp-tail-retain-matched.jsonl`。这增强了生命周期/异步组合问题的证据；单独进程的allocator/任务队列历史仍与前三分布连续运行不同，未证明具体根因或完成绑定修复。

现有RunOpApiV2绑定尚未达到完整异步组合门禁。证据提示应检查生产者任务顺序、中间张量storage/stream生命周期和工作空间；未证明根因，不能归咎310P不支持INT4，也不能声称原算子尾块已损坏。

## 复现

先按上一轮 `results/qwen36/mindieq-mlp-20261013/start-candidate.sh` 配置外部固定制品；诊断启动前确保芯片0没有其他服务。环境变量只添加：

```bash
QWEN36_ROUTE_TRACE_PATH=/tmp/route-occupancy.jsonl bash start-candidate.sh
```

服务health就绪后在a2运行：

```bash
python3 /project/tools/collect_route_occupancy.py --out /work/route-occupancy
cat /work/route-occupancy/technical.jsonl /work/route-occupancy/meeting.jsonl /work/route-occupancy/code.jsonl > /work/route-occupancy/combined.jsonl
python3 /project/tools/summarize_route_occupancy.py /work/route-occupancy/combined.jsonl --out /work/route-occupancy/summary-new.json
```

`collect_route_occupancy.py`是在a2宿主执行，`/project`/`/work`须替换为宿主仓库和工作目录。其podman exec touch/rm标记、复制采样文件，不运行独立NPU进程。输出目录必须没有同名结果。

停止服务后顺序运行本目录`run-full-mlp-*.sh`，禁止与服务并发占用davinci0。其中脚本保留此次a2绝对路径，均使用已编译独立MindIEQ任务队列绑定。需要上一轮外部镜像/OPP/扩展；不是单靠仓库即可运行。

## 工程决定

保持TP1/48G/112K、W4A8+C8+MTP2+Graph+GDN、4096/512、prefix关闭的部署候选。本轮未改变推理默认值，也未生成新模型权重。继续投入方向是先解决MindIEQ异步完整MLP数值门禁，再做设备端按专家行数选择或融合GMM，避免每层读取CPU counts/逐专家提交。再次通过完整MLP、Graph、模型质量后才开展端到端集成。

实验结束后已关闭路由诊断并恢复a2端口18024常规候选，health HTTP200；英语/中文/代码/数学/重复文本5项64-token生成检查PASS（candidate-restored-sanity.json）。这是基本生成检查，不是完整质量评估或本轮长上下文复测。
