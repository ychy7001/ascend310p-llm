# a1 W8A8 TP2部署参考（2026-10-14）

只读调查现有服务并执行HTTP测试，没有停止服务、修改配置或权重。a1 API18081，Qwen3.6-35B-A3B-w8a8；TP2/FP16 KV（auto）/Graph FULL_DECODE_ONLY[1]/262144/.9/maxseq1/batch4096/prefix开启；启动参数没有speculative-config。镜像、驱动26.0.rc1、CANN9.1.0-beta.1及过滤后的配置见runtime.json。

使用bench_server_phases.py相同8K/32256输入和128输出、warmup1/repeat3，Prompt token SHA与a2 E5逐一匹配。请求期间phase计数严格一条。首次unsalted输出保留，但32K可以复用此前8K前缀，不作为冷Prefill结论。正式cold-*每请求独立cache_salt，已确认本版本CompletionRequest支持；逐请求记录prefix查询/命中counter，三次命中均0。

| 输入 | a1 W8A8 TP2 Prefill/Decode tok/s | a2 E5 W4A8 TP1/C8/MTP2 Prefill/Decode tok/s |
|---|---:|---:|
|8192|1298.6 /31.95|624.8 /34.68|
|32256|1094.3 /29.16|546.6 /28.96|

a1当前证实Prefill1K+；TP2与TP1、两台机器的软件/驱动、KV精度和MTP等均不同，不能把表中差异归因于W4量化或作为同机性能验收。a1已承担其他服务，测试无重启且没有默认更改。较长输入两个芯片参考约为TP1两倍，仅是这两组测量现象，不是线性扩展上界。

复现须对每次新运行选全新salt前缀，避免重复调用同一salt重新命中。以下示例需替换后缀，输出文件不能覆盖已有结果：

```bash
python3 tools/bench_server_phases.py --base-url http://127.0.0.1:18081 --model Qwen3.6-35B-A3B-w8a8 --prompt-tokens 8192 --output-tokens 128 --repeats 3 --unique-prompts --cache-salt-prefix YOUR_UNIQUE_RUN-p8192 --out /tmp/cold-p8192.jsonl
```

正式速度取comparison.json；完整原始响应、usage、流式事件、输入SHA、phase及prefix counters均在cold-*jsonl。未测a1 MTP、多并发、长上下文质量或实际256K请求。
