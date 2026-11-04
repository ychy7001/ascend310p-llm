# Ascend 310P 基线身份与测量协议

## 生产配置身份

生产参照为Qwen3.6-35B-A3B W8A8 TP2：MTP0、Graph[1,2]、prefix on、maxlen262144、batch4096、seqs4、FP16 SSM、async，量化后端ascend。

镜像digest为`sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38`；vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，Ascend `88eac2713b14beedc7a48803274f8d6cee3e9f30`，torch-npu2.10.0.post2，CANN9.1.0-beta.1。a1生产驱动26.0.rc1，a2隔离副本驱动26.1.1；副本测量不直接等同在线性能。

基线数字、权重校验范围、W4预算支线与候选比较统一见[结果](ascend310p-results.md)。27B Dense使用独立基线；模型名、量化、TP、软件栈和上下文任一变化都需单独标识。

## 测量协议

- 冻结镜像digest、源码/库/权重/tokenizer哈希、有效参数、输入token数组及哈希、seed/采样、输出上限、warm/cold与cache状态。启动及权重加载时间单独记录。
- 冷Prefill以C1、1K/4K/16K/32K输入、256固定输出起步：seed0、temperature0、ignore_eos、逐请求独立cache_salt，并核对usage、phase计数及prefix hits增量0。每档3次只作筛选，不报告可靠P95。
- 并发以1K/256的C1/C3/C4聚合输出中位数比较，分母为批次墙钟、包含Prefill。继续扩展前先估算状态/KV容量；内存采样注明时间窗、间隔及指标来源，不把设备采样峰值称为精确HBM峰值。
- 每请求保存状态、usage、TTFT、mean TPOT、E2E、失败记录；另记批次吞吐、MTP计数及样本数。SSE首个非空文本块用于TTFT；MTP可能每块多token，块间延迟不直接作逐token ITL。客户端TPOT与服务端阶段时间分报。
- 正确性覆盖前后短题、A/B/A隔离与canary、事实/schema、原文纪要语义和长后短。GSM使用固定64索引、zero-shot、thinking off、greedy、512输出上限；截断与解析缺失原样计入。1024上限诊断另归档。
- 性能用A/B/A′检查时间漂移；合成业务回放固定token形状、输出长度、seed、阶段依赖与并发，只计业务形状墙钟。原文质量及字段配对另验，见[配对协议](research/action-state-paired-validation.md)。
- Profile与普通计时分开；记录HTTP捕获边界、rank、导出及可解析性，采集后恢复普通计时和正确性检查，见[Profiling](ascend310p-profiling.md)。

## 工具使用

- `tools/bench_http_stream.py`：记录SSE与聚合吞吐；main复用prompt且不传salt，prefix开启时warmup/重复命中会污染冷Prefill，冷测须显式隔离。
- `tools/bench_server_phases.py`：独占端点阶段counter差值、unique prompt、salt、零命中与MTP计数。重叠请求的全局counter差值不作单请求耗时。
- `tools/replay_stage_workload.py`：固定形状、seed、阶段并发与salt，用于业务墙钟对比。
