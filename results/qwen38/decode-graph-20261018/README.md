# Qwen3.8 TP2 Decode Graph 独立实验（2026-10-18）

基线：[完整 TP2 文本服务与 4K 实测](../tp2-serving-20261018/README.md)，代码 a823d28。
本实验在同一台 a2、单张 300I DUO 的两颗 310P3 上进行，沿用相同的完整检查点、固定源码、运行时镜像和三个基线补丁。

## 固定条件与唯一变量

- TP2/EP2、Routed Experts 原生 G128 W4A8，其余层 FP16，C16、MTP0。
- `max_model_len=4096`、`max_num_seqs=1`、`max_num_batched_tokens=256`、`gpu_memory_utilization=0.95`。
- Prefix Cache 关闭、`mamba_cache_mode=none`、HCCL PG buffer 64 MiB。
- eager 基线改为 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，移除 `--enforce-eager`，设置 `mode=0`、`cudagraph_mode=FULL_DECODE_ONLY`、`cudagraph_capture_sizes=[1]`。
- 开启 `--cudagraph-metrics`；不启用 MTP，不增加并发，不改动权重文件。

此处的 Graph 是可中断的 ACL Graph：PLE 主机查表/传输以及需要动态选择的 QSA 操作在图间执行。它不表示全部运算均在一个设备图中，也不表示启用了 torch.compile。

## 复现与回滚

a2 工作目录为 `/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018`，模型及源码路径与基线一致。运行前先复制本目录的三个脚本到工作目录。

```bash
# 在 a2 上切换；先保留原来的 eager 容器，便于回滚。
podman stop --time 20 qwen38-tp2-text-4k
bash /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018/launch-text-tp2-graph-on-a2.sh
python3 /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018/run-graph-gates.py

# 回滚到已保留的 eager 容器；两个容器不能同时占用两颗 NPU。
podman stop --time 20 qwen38-tp2-text-4k-graph
podman start qwen38-tp2-text-4k
```

端口仍为 `18038`，API 模型名仍为 `qwen38-w4a8-tp2-text`。原始 inspect 含环境信息，仅保存在 a2，不公开。

`run-graph-gates.py` 等待健康检查后执行三条基本生成检查，再对 512/128、3968/128 分别先预热 32 个输出 token，随后执行三次计时请求。保留每个 SSE 事件和实际 usage，强制校验输入/输出 token 数。

计时口径与基线一致：Decode 为 `(output_tokens-1)/(最终 usage 时刻-首个非空输出时刻)`；输入数除以 TTFT 仅代表包含 HTTP、调度和首 token 的有效速率，不是纯 Prefill Kernel 吞吐。三次样本不足以确定产品 P95。基本生成检查也不能替代模型精度评估。

## 实际结果

**通过本轮 4K、单流门禁，保留 Graph 服务运行。** 容器为 `qwen38-tp2-text-4k-graph`，原 eager 容器保留但停止；未触碰本机 3090 的 Strata。

| 输入 / 输出 token | eager TTFT | Graph TTFT | eager Decode | Graph Decode | Decode 中位变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 128，各 3 次 | 1.909–1.911 s | 1.904–1.992 s | 13.82–13.89 tok/s | 16.50–16.72 tok/s | **+20.14%** |
| 3968 / 128，eager 1 次、Graph 3 次 | 15.841 s | 16.137–16.239 s | 12.68 tok/s | 16.26–16.35 tok/s | **+28.55%，初步对照** |

输入 token 除以 TTFT 的中位数：512 档从 268.04 降为 258.52 tok/s，3968 档从 250.49 降为 245.01 tok/s。TTFT 中位数分别增加 3.68% / 2.24%。Prefill 本轮没有收益，不应将 Decode 加速写成 Prefill 加速。4K eager 只有历史单次样本，因此 28.55% 需要后续交替多轮复测，不能当作正式稳定增益；512 档双方各三次已显示约 20% 的重复收益。

真实生成与状态刷新：

- [基本生成](graph-smoke.json) 3/3 通过：42、中文解释、READY。
- [交替请求](graph-request-refresh.json) 8/8 通过：两道算术、六个独立口令，实际输入长度依次为 25 / 24 / 32 / 533 / 43 / 3033 / 32 / 133 token，最长请求后再次检查短请求，全部精确返回期望答案。
- 两组 Benchmark 各自预热，6 个计时请求的 usage 全部精确匹配；3 次 `3968+128=4096` 实际请求均完整完成。
- 512 档 eager 与 Graph 的文本哈希集合一致，两边各出现两个输出版本；3968 档文本哈希未一致。原始文本全部保留。这组重复性观察和 11 个简单闭式答案检查均不能证明整体模型质量等价，仍需完整质量集或 teacher-forced logits 比较。

图实际执行证据：

- 启动日志明确显示 `Capturing CUDA graphs (decode, FULL)`，捕获 1 个形状。
- [运行模式汇总](graph-runtime-mode-counts.json)记录周期日志中的 883 次 `FULL`、96 次 `NONE`；`FULL` 均为单 token Decode，较大 Prefill 块为 `NONE`。模式统计与成功捕获、真实输出共同构成本轮执行证据。
- 两个 rank 的图内存日志分别约 0.18 / 0.19 GiB，权重仍为 37.26 GiB/rank，torch allocated 约 38.24 GiB/rank。
- [捕获后资源采样](graph-resource-summary.json)共 40 次：chip0 最大使用 `42345/44277 MiB`、最少空闲 `1932 MiB`；chip1 最大使用 `41823/43693 MiB`、最少空闲 `1870 MiB`。eager 前一轮最少空闲为 1995 / 1947 MiB；两轮采样差约 63 / 77 MiB，不能将这项总占用差直接当作图池的独立大小。
- 物理 attention block 数仍为较小 rank 的 578，单请求 Mamba pool 仍为 57,729,024 字节。日志的约 10.36 GiB 是 planner 逻辑预算，不是额外分配了 10.36 GiB 物理显存；实际验证上限仍为 4096。

## 工程决定与后续优先级

本轮仅需图配置，不增加任何 Python/C++ 算法补丁。对于**当前 Qwen3.8、TP2/EP2、C16、MTP0、单请求、4K 实验档**，推荐保留 Decode Graph；完整产品规格仍需质量、并发与长上下文验证，不能沿用 Qwen3.6 的 112K 或 MTP/C8 结论。

下一步按收益和风险排序：先在此固定配置上逐档探索 8K/16K 的真实检索与生成、内存和 TTFT/Decode；再单独对照 PLE/LM Head 动态 W8A8 投影，以判断能否腾出有价值的权重内存并提速；最后评估 MTP（draft 权重与额外状态是否能装下）和 C8（Qwen3.8 QSA 的存储/计算协议需独立适配）。Prefix Cache 需要解决最少 64 状态槽造成的容量压力后再启用。

原始记录包括 [512/128](graph-bench-512-128.jsonl)、[3968/128](graph-bench-3968-128.jsonl)、[完整日志](graph-serving-final.log.gz)、[原始资源采样](graph-resources.jsonl.gz)。

汇总可重算：

```bash
python3 tools/qwen38_graph_compare.py \
  --eager results/qwen38/tp2-serving-20261018/serving-bench-compact-512-128.jsonl \
  --graph results/qwen38/decode-graph-20261018/graph-bench-512-128.jsonl \
  --output results/qwen38/decode-graph-20261018/comparison-512.json

# 在 a2 上、无其他请求时执行闭式答案/长短请求刷新检查。
python3 /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018/qwen38_graph_request_gate.py \
  --output /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018/graph-request-refresh.json
```

Benchmark 文件以 `open('x')` 创建，重跑时需使用新的输出文件名，避免覆盖原始结果。`run-graph-gates.py` 同理仅用于新实验目录或尚无结果的首次运行。
