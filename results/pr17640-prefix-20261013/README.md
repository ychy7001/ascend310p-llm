# 新GDN候选的prefix复用验收

2026-10-13，a2芯片0；TP1/48G、N128 W4A8、动态C8、MTP2、FULL_DECODE_ONLY mode0/capture1,2,3、maxlen114688、.95、调度4096/专家512、maxseq1。仅显式开启prefix cache；框架同时自动选mamba align、逻辑block1152。冷一次后热三次，正常EOS、输出上限128，完整输入IDs及输出/usage/流事件保留。

| 输入tokens | 冷TTFT秒 | 热TTFT中位秒 | 每条热请求命中tokens | 输出 |
|---:|---:|---:|---:|---|
| 8192 | 15.839 | 2.646 | 6912 | 四条全文一致 |
| 32256 | 68.470 | 11.137 | 27648 | 四条全文一致 |

TTFT分别降低83.29%/83.73%。热请求跳过已缓存计算，logical_prefill_tok_s不能视为实际冷Prefill算力。两长度Decode基本不变，见comparison.json。这是同服务一次冷对三次热，不是重复冷基准或P95。

共享文档A-cold/A-hot/B-shared/A-after-B/B-cold检查通过：A问秘密码，B问负责人；两个答案正确，A三条文本一致，B共享/独立冷参考全文一致。`cache_salt`隔离建立B冷参考，指标可核实其命中为0。`state8192-final.jsonl`用最终答案严格匹配，不能以思考段出现答案作为通过。

首次调用reset_prefix_cache接口404、该空结果不作为数据。原版B输出上限32截断，保留state8192-salted.jsonl用于解释失败；随后B上限256正常结束，保留state8192-salted-o256.jsonl。最终校验器改为严格检查</think>之后答案，并重跑state8192-final.jsonl。

启动：按docs/gdn-112k-candidate.md提供制品路径，设置ENABLE_PREFIX_CACHING=1、CONTAINER_NAME=qwen36-pr17640-prefix，执行GDN候选启动器；精确路径见start.sh。默认prefix off保持不变。

复现：

```bash
python3 tools/bench_prefix_reuse.py --base-url http://127.0.0.1:18024 \
  --model qwen36-w4a8-c8 --prompt-tokens 8192 --output-tokens 128 \
  --repeats 3 --out /tmp/reuse8192.json
python3 tools/bench_prefix_reuse.py --base-url http://127.0.0.1:18024 \
  --model qwen36-w4a8-c8 --prompt-tokens 32256 --output-tokens 128 \
  --repeats 3 --out /tmp/reuse32256.json
python3 tools/check_prefix_state.py --base-url http://127.0.0.1:18024 \
  --model qwen36-w4a8-c8 --prefix-tokens 8192 --out /tmp/state8192.jsonl
```

开启prefix的启动预算：日志报告KV池181248tokens（原候选186236），自定义C8分配116块、单请求预算最少113块，payload约4.04GiB、scratch224MiB，Graph.31GiB。池估计不是单请求上限。实际114560token检索及后续状态复用尚在执行，未据小请求结果改变默认。

实际114560token检索通过，正确输出ASCEND732941、总114574tokens、耗时348.211s，关闭prefix同类单样本348.941s。之后A/B/A全部最终答案正确；A三条全文一致，B共享/冷参考的思考段出现少量措辞差异，最终答案LinMei、两条均232输出tokens。严格B全文一致检查为false，未隐藏。补测同一B冷参考的热请求两次全文完全匹配；另一独立cache_salt冷请求全文也有变化、最终答案相同。因此目前不能将变化单独归因于prefix，但也不宣称严格等价。保留prefix为可选试验，默认off。服务18024暂保留prefix候选用于后续诊断。

服务日志归档只归一化换行及行末空白，a2原始log保留。该轮没有证明全面长文质量、最长上下文、并发批处理或P95。
