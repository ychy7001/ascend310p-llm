# MindIEQ输入规模分流（D4，2026-10-14）

基于C3长输入受益、短输入TTFT增加70～80ms，引入QWEN36_MINDIEQ_MIN_TOKENS。历史默认17保持不变，D4设128；仅当进入MoE的block_input行数达到阈值时采用MindIEQ。它可能包含调度padding，不能解释成用户prompt长度或累计上下文阈值。阈值至少17，Decode仍native；总开关默认off。

D4沿用C3全部配置与新A8库，TP1/48G/W4A8/C8/MTP2/Decode Graph/GDN/4096:512/112K/.95/maxseq1/prefix off，权重和依赖不变。源代码/库hash在artifacts.json，启动前preflight记录阈值128/custom_quant1及库hash。业务和8K复测、两题诊断已完成；完整64题复测正在执行，尚不判定全部分流门禁通过。

start-service.sh与run-service-gates.sh为本轮实验机命令；需替换work/code/release路径。后者先health等待，再sanity5例、4类业务warmup1/repeat3、8K/128warmup1/repeat3及spec counters、第711/205题同512上限诊断。

```bash
python3 results/mindieq-input-dispatch-20261014/analyze.py
python3 tools/check_source_snapshot.py docs/source-snapshot-20261013.json
```

分析脚本要求全部输出完整，核对C3和历史native业务提示词SHA及8K输入SHA。D4/C3为连续独立启动，历史native并非本轮交替重启；不能用这组比较证明全部部署条件下稳定收益。边界两题也不能替代此前64题或全面模型精度验收。

## 已完成结果

| 场景 | C3阈值17 TTFT秒 | D4阈值128 TTFT秒 | D4 Decode tok/s |
|---|---:|---:|---:|
| 对话/45输入 |0.466|0.393|31.628|
| 会议/962输入 |1.921|1.921|39.180|
| 代码/30输入 |0.408|0.337|39.002|
| 分析/42输入 |0.464|0.386|33.948|

短输入额外开销消除至历史native水平，会议收益保留。三类短场景均触及256输出上限，会议自然结束；文本存在差异，不是业务质量评分。

8K三次输入SHA与C3/A1逐一匹配，D4 Prefill中位607.457 tok/s、Decode34.821 tok/s、MTP接受率92.963%，与C3 607.019/34.734/92.963%接近。相对历史A1 502.936/37.038仍约+20.8% Prefill、-6.0% Decode。这轮独立重启支持长输入收益重复，但不是随机交替A/B，不能宣称模型无数值差异。

两题诊断711正确6、自然结束；205仍共同512上限截断。基础sanity5/5。最初诊断CLI索引误用空格导致请求前退出，原失败日志保留，修正逗号分隔后补跑；未重启服务或重复已完成性能测试。

下一步完整64题复测已启动，远端PID3996511，gsm64.jsonl/gsm64.txt；本提交不计入尚未完成的结果。当前可选部署阈值128适合作为候选，默认17及总开关off保持不变。剩余长输入MTP接受率差异需独立定位，不能靠提高阈值回避长输入收益与精度目标。

metrics-final.txt/npu-smi.txt为业务、8K及边界题结束后的记录，采集于64题复测前；state.txt表示发布时仍在gsm64阶段，不作为进程存活证据。
