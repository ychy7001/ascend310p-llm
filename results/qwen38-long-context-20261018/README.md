# Qwen3.8 TP2 / Decode Graph 长上下文实测（2026-10-18）

沿用[已通过的 4K Graph 配置](../qwen38-decode-graph-20261018/README.md)，仅将 `max_model_len` 从 4096 提高为 16384。固定 TP2/EP2、G128 W4A8、C16、MTP0、Prefix关闭、batch256、maxseq1、memory0.95、PG64MiB、breakable FULL_DECODE_ONLY/capture[1]，权重和源代码未修改。

容器 `qwen38-tp2-text-16k-graph`；端口18038，模型名 `qwen38-w4a8-tp2-text`。原 4K Graph 和 eager 容器均保留停止，不可同时启动占用相同两颗 NPU。

## 门禁协议

完整权重重新加载，健康检查后基本生成3题；8064/128和16256/128各先预热32输出token，再执行3次计时，严格校验usage。随后执行16256输入token、中部唯一口令检索（最多64输出token），并再次做8条交替长短请求检查。

计时口径与4K报告一致：Decode按首个非空输出至最终usage的时间和后续token数计算；输入数/TTFT不是纯kernel Prefill吞吐。测试输入是固定的重复文本；PLE主机页缓存命中和业务自然文本可能不同。检索是单一canary，并不是完整长文理解评估。

## 实际结果

**16K / batch256 全部门禁通过。** 实际完成三次 `16256+128=16384` token请求；这证明本配置可处理16K请求，不代表硬件最大上下文，也不是对所有16K任务的质量保证。为随后独立的[batch512对照](../qwen38-prefill-batch512-20261018/README.md)，本容器已停止并保留，可按下方命令回滚。

| 输入 / 输出 token | 计时次数 | TTFT 范围 | Decode 范围 | Decode 中位 | 输入数/TTFT 中位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8064 / 128 | 3 | 33.671–33.866 s | 16.17–16.20 tok/s | 16.19 tok/s | 238.66 tok/s |
| 16256 / 128 | 3 | 69.184–69.299 s | 15.56–15.60 tok/s | 15.59 tok/s | 234.87 tok/s |

固定批次下，从8K到16K，Decode中位下降约3.72%，输入数/TTFT下降约1.59%；Prefill是绝对耗时的大头。与前一轮4K Graph约16.30 tok/s相比，16K Decode下降约4.35%，但这不是同一次随机交替实验。

质量与状态门禁：

- [基本生成](long16k-smoke.json)3/3通过。
- [16256-token中部检索](long16k-needle-16256.jsonl)：口令位于token8120，正确返回 `ASCEND732941`，实际输出14token、总长16270、耗时72.19s。
- [长请求后的交替请求](long16k-request-refresh.json)8/8精确匹配，包括3033-token请求后再次检查短请求。
- [原始8K计时](long16k-bench-8064-128.jsonl)、[原始16K计时](long16k-bench-16256-128.jsonl)及[统计汇总](benchmark-summary.json)包含预热和usage；计时样本没有把预热算进去。
- 全部为少量合成输入/闭式答案检查，不代表自然长文理解、完整精度评估、并发或正式P95。

日志仍显示权重37.26GiB/rank、图内存约0.18/0.20GiB、物理attention最小578blocks；planner逻辑预算约10.36GiB，不能把它当成物理内存。原始[服务日志](long16k-serving-final.log.gz)、[最终NPU快照](long16k-npu-final.txt)、[门禁状态](long16k-gate-status.json)已保存。资源采样将随batch512对照统一收集并按运行阶段分开统计。

## 复现与回滚

将本目录的两个shell脚本及run-long16k-gates.py复制到a2工作目录 `/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018`；该目录需已有基线工具 `bench_http_stream.py`、`qwen38_serving_smoke.py`、`check_long_context_needle.py`、`qwen38_graph_request_gate.py`。

```bash
podman stop --time 20 qwen38-tp2-text-4k-graph
bash /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018/launch-text-tp2-graph-16k-on-a2.sh
python3 /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018/run-long16k-gates.py

# 回滚4K Graph
podman stop --time 20 qwen38-tp2-text-16k-graph
podman start qwen38-tp2-text-4k-graph
```

如需重跑必须使用新的结果文件名或独立目录；基准工具拒绝覆盖原始jsonl。checkpoint校验及GDN门禁仍由launcher执行，inspect只留a2，不公开环境快照。
