# GDN / batch4096 / MTP1与2对照

a2芯片0，2026-10-13；TP1、W4A8、动态C8、Decode Graph、112K、利用率.95、调度4096、专家chunk512、max_seqs1、prefix off。

每类chat预热1次、重复3次；temperature0、thinking关闭、输出上限256。先执行MTP2再MTP1。原始jsonl包含全文、流事件、usage、阶段计数及MTP计数差分；`comparison.json`汇总中位数。不是任务质量评分。

复现测试（须先按`docs/gdn-112k-candidate.md`独占芯片启动对应MTP服务）：

```bash
python3 tools/bench_mtp_chat.py --base-url http://127.0.0.1:18024 \
  --label gdn-b4096-mtp2 --out /tmp/mtp2-chat.jsonl
# 停止该服务，QWEN36_PROFILE=mtp1启动同参数服务后执行：
python3 tools/bench_mtp_chat.py --base-url http://127.0.0.1:18024 \
  --label gdn-b4096-mtp1 --out /tmp/mtp1-chat.jsonl
python3 tools/bench_server_phases.py --base-url http://127.0.0.1:18024 \
  --model qwen36-w4a8-c8 --prompt-tokens 8192 --output-tokens 128 --repeats 3 --unique-prompts \
  --out /tmp/mtp1-p8192-o128.jsonl
```

MTP1精确启动环境见`start-mtp1.sh`，MTP2见`../pr17640-b4096-20261013/start.sh`。MTP2的8K对照直接引用该目录原始数据，双方逐条prompt token SHA256一致、三条输出一致。

会议摘要正常结束且重复/跨配置文本一致，其余三类均达到256token上限、出现重复文本变化；完整任务质量待更长输出及更大测试集评估。两配置都观察到文本变化，尚未定位到特定算子。不要把小样本速度比较作为精度验收。

保留MTP2为实验默认：8K Decode约+21.36%，四类chat约+1.66%至19.01%；8K Prefill约-1.47%，四类chat TTFT高43–65ms，完整请求耗时均较低。MTP0、prefix caching、并发与尾延迟待测。启动KV池估计不代表最大上下文。

归档服务日志仅归一化换行并去掉行末空白；a2原始`.log`保留。
