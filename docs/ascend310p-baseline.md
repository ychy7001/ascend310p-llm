# Ascend 310P 基线状态

2026-11-03：首次只读审计已完成；用户同意继续后，已启动隔离基线实验，下方增量持续记录。环境与功能矩阵见[计划](ascend310p-optimization-plan.md)。

## 生产基线身份（用户确认）

a1生产实际运行W8A8 TP2 / MTP0 / Graph[1,2] / prefix on / maxlen262144 / batch4096 / seqs4 / FP16 SSM / async。生产镜像digest为`sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38`。vLLM `0fc695fc…`，Ascend `88eac271…`，torch-npu2.10.0.post2，CANN9.1.0-beta.1，a1驱动26.0.rc1。

只读审计确认服务运行、Prefix Cache开关已启用。一次metrics快照中的prefix queries3768、hits0只代表当时进程累计计数，不能推断缓存失效或真实业务命中率。a1未执行新Benchmark。a2已复制并校验生产权重，完成隔离基线、业务回放及双rank Profile；a2驱动26.1.1与a1不同，不冒充生产原机复测。

下方TP1 W4结果归入预算方案。其与生产不同量化/TP/版本/上下文/MTP，不能直接作为生产优化收益比较。

## W4预算方案的可复现起点

优先恢复到独立容器：a2保留镜像 `b381d8e5ed45…`，W4A8 TP1 / C16 / MTP2 / MindIEQ / Decode Graph / 114688上下文 / batch2048 / seqs1 / prefix off / async。这只是保存的配置，需先通过启动与正确性门禁才能成为新基线。

W8A8 TP2 Graph MTP0是10月31日业务推荐，尚需定位a2可用权重与精确镜像，另立基线，不能把W4和W8变更混作单变量优化。

## 历史参考

| 历史实验 | 结果 | 边界 |
|---|---|---|
| W4 MTP2 vs MTP1，112K Graph、32K/256 | Decode23.20→28.94 tok/s | 特定历史单流协议，非今日基线 |
| MindIEQ vs native W4 | Prefill8K +23.29%、32K +20.69% | 历史交替五次中位数，非新并发验收 |
| 10月31日W8 TP2 Graph MTP0 | 长会议509s、两场943s、短课堂209s | 业务阶段回放墙钟，不是单流Decode |
| 同轮W8 eager MTP2 | 478s / 838s / 212s | 出现职责错配，不能推荐 |
| GSM64 E5/D4 | 63/64，数字答案无变化 | 小子集，含共同截断；非完整精度等价 |

原始链接分别见 qwen36/README.md、mindieq-evaluation-20261014.md、business-replay-20261031.md、quality.md。

## 基准工具检查

- `tools/bench_http_stream.py`：固定token输入、固定输出、warmup/逐请求SSE与聚合吞吐已有。main复用同一prompt，未传cache_salt；prefix开启时warmup和请求间命中会污染冷Prefill。底层request支持salt，后续优先复用。
- `tools/bench_server_phases.py`：有独占端点的阶段counter差值、prompt哈希、unique prompt、每请求salt、零prefix hit检查和MTP计数；不适合把重叠请求的全局counter差值当单请求耗时。
- `tools/replay_stage_workload.py`：固定token形状、seed、阶段并发及salt，适合业务墙钟对比；合成回放不替代原文质量集。
- SSE记录的是首个非空文本块；MTP可能一次发送多token，块间延迟不能直接当逐token ITL。客户端mean TPOT与服务端阶段时间应分开报告。
- 现有stream脚本遇到异常会raise，需要补齐失败请求记录、成功率和分位数汇总；三轮请求只能作筛选，不能称为可靠P95。

下一轮记录：输入token数组与哈希、seed/采样、完整有效配置、镜像digest/源码/库哈希、warm/cold/cache状态、每请求usage/状态/TTFT/mean TPOT/E2E、批次墙钟吞吐、MTP计数、内存采样时间窗和峰值、统计样本数。先C1全长度，再按容量升C4/16/32，禁止直接铺满笛卡尔积。

## 2026-11-03 隔离基线执行增量

镜像168个文件哈希与基础源码提交校验通过，独立容器qwen36-baseline-20261103，127.0.0.1:18103。固定参数与启动入口在benchmarks/ascend310p/。启动时间（含权重加载、视觉预热与Graph准备）单独记录，不计入正式吞吐。首次8题严格8/8，A/B/A全文一致、无canary泄漏。

四档每档正式3次均usage匹配、phase计数有效且prefix hits增量0；seed0、temperature0、ignore_eos、固定256输出，输入数组/哈希保存。3次仅供筛选。

| 输入 | Prefill中位tok/s | Decode中位tok/s | TTFT中位s | mean TPOT中位ms | 成功 |
|---|---:|---:|---:|---:|---:|
| 1024 | 590.35 | 34.93 | 1.738 | 28.627 | 3/3 |
| 4096 | 630.80 | 33.71 | 6.497 | 29.661 | 3/3 |
| 16384 | 594.65 | 31.96 | 27.562 | 31.269 | 3/3 |
| 32768 | 543.88 | 28.87 | 60.265 | 34.598 | 3/3 |

长后短8/8及A/B/A检查也通过。GSM64已完成64/64正确、无答案解析缺失、无截断；这是seed310的固定小子集，不能据此宣称与原模型精度等价。52个分片全部SHA256通过（93125个张量，其中92160个expert张量）。原始数据：results/baseline-20261103/；远端完整目录/path/to/lab-runtime/hyperqwen-20261103。

32K正式请求Prefill约60秒、Decode约9秒，说明本请求形状下Prefill主导E2E；这不是纯kernel瓶颈结论。可优先验证共享长前缀的Prefix Cache单变量候选，但冷/热请求必须分报，命中后的logical prefill tok/s不当作冷Prefill算力。

资源采样见results/baseline-20261103/resource-summary.json：设备0 Memory-Usage采样最大值39153 MB，AICore采样最大值100%。这是npu-smi设备内存指标，不能当作精确HBM峰值；采样最大间隔5.64秒。

## W8A8 TP2生产配置副本：首轮性能（a2）

同生产镜像和W8权重，262144上下文、MTP0、Graph[1,2]、seqs4、batch4096、prefix on、FP16 SSM。Profiler已配置但常规窗口未启动；a2驱动26.1.1不同于a1的26.0.rc1。权重10分片、93951张量，39,810,885,759字节；index/边界检查通过，a2所有文件SHA指纹已保存，与a1元数据SHA及全文件长度一致。权重经SSH/rsync复制，未独立在a1重读全权重做SHA对比。

四档正式3次均有效，独立salt、prefix hits增量0，256固定输出、seed0。3次仅筛选，不当作可靠P95。

| 输入 | Prefill中位tok/s | Decode中位tok/s | TTFT中位s | E2E中位s | 成功 |
|---|---:|---:|---:|---:|---:|
| 1024 | 973.65 | 32.90 | 1.054 | 8.804 | 3/3 |
| 4096 | 1290.43 | 32.44 | 3.178 | 11.028 | 3/3 |
| 16384 | 1204.27 | 30.86 | 13.615 | 21.871 | 3/3 |
| 32768 | 1082.54 | 29.12 | 30.286 | 39.037 | 3/3 |

另用固定1024/256 stream协议，每档3轮；并发1/3/4聚合输出中位分别28.97 / 48.86 / 63.04 tok/s，请求3/3、9/9、12/12成功。这里聚合吞吐包括Prefill与请求墙钟，不能与上表Decode速率混比。完整原始与汇总见results/production-w8a8-20261103/。前后8题/A-B-A通过，业务28条自动完成/结构/工具检查通过；语义需另审，尚不能称完整业务质量通过。

## W8 TP2完整业务回放

固定`chinese_company_annual_meeting_planning`合成回放，按真实阶段调用长度与依赖关系、全局并发3执行；26/26成功，输入合计76631 tokens、输出20690 tokens，墙钟508.4s，含Prefill的聚合输出40.7 tok/s。独立run/request salt，整个回放缓存命中增量0；这是冷缓存对照，生产缓存仍开启，不表示线上必定没有命中。原始每请求usage/时间见results/production-w8a8-20261103/business-replay-long.jsonl。

该回放固定输出长度，用于业务形状的计时，不是纪要内容质量评测，也不包含实际私有会议文本。事实/schema门禁与纪要语义对照单独记录；当前只执行一轮，不能评价尾延迟或宣称稳定收益。

常规32K C1阶段`npu-smi Memory-Usage`采样峰值：device0为39138MB/44277MB，device1为38435MB/43693MB；27个阶段样本，约5秒间隔加查询开销。仅代表设备采样峰值，非精确HBM/allocator峰值，详见resource-summary-initial.json和原始采样。

W8 TP2同一固定64题、zero-shot、thinking off、greedy、输出上限512的GSM结果为63/64，parse_missing=1、truncated=1。失败索引205在512 tokens处终止，缺失`####`最终答案；不能把未完成推导直接判为数值错误，也不能补算为正确。完整原文保存在gsm64.jsonl。W4的64/64是不同量化/执行配置，不能借用。候选按同协议比较，若另做1024上限诊断，需单独归档。


## Dense 基线边界更新

27B Dense 310P W8A8权重已校验并实际尝试TP2启动；生产栈在首次profiling的动态量化MLP down_proj报507015，独立同步eager及ACLNN_CACHE_LIMIT=0均未解决。尚无Dense可报告的请求性能/正确率，不将35B-A3B或W4支线指标套用为Dense基线。详见ascend310p-results.md及results/qwen38-dense-baseline-20261103。
