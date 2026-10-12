# 质量与缓存回退验证协议

有限生成sanity不是完整精度验收。原始响应、usage、截断原因、逐题结果和数据集留在实验目录，不进入Git。

## 已测请求复用小样本

experimentally verified：TP1/W4A8/C8/eager/MTP=0/4K，8个算术/排序/提取/中文题严格答案8/8；A/B/A中间3072输入+64输出干扰，前后答案文本一致、未泄漏标记。脚本`tools/check_cache_request_reuse.py`，原始`c8-reuse-quality.json`。不能推广为正式准确率或全部allocator隔离。

该脚本现在记录前后spec_decode Prometheus指标，供MTP候选检验实际draft/accept/reject覆盖；计数器变化与服务生成结果需一起看，不能把health或MTP配置存在当成有效verify证据。

## 跨页拒绝槽覆盖探针

`experimental/check_c8_rejected_slots_310.py --library <libqwen36_c8.so>`对prefix63/64/127写入两枚错误draft的K/V，再限制有效长度读取，要求与写入前完全一致；覆盖下一提交位置，和新建干净C8缓存的原生PA对照。测试包括scale覆盖和页边界。探针尚待真实设备执行；即使通过，也只证明存储/PA语义，不替代scheduler及linear state rollback。

## GSM8K固定子集协议

数据来自[官方grade-school-math](https://github.com/openai/grade-school-math)，固定commit `3101c7d5072418e28b9008a6636bde82a006892c`、文件`grade_school_math/data/test.jsonl`。实际1319行，SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`；经GitHub API下载，不重新分发数据。

首轮32题、random seed310、选中indices排序后顺序执行；temperature0、enable_thinking=false、max_tokens512、固定zero-shot后缀，严格提取最后的`####`数字。保存全部响应，未解析或截断不自动当正确，不用最后一个随意数字兜底。此协议不是标准lm-eval完整GSM8K分数；32题只适合早期回归定位，不足以证明小幅准确率变化。

```bash
python3 tools/eval_gsm8k_http.py --base-url http://127.0.0.1:18016 \
  --model qwen36-w4a8-c8 --label c8-mtp1 --dataset /work/gsm8k-test.jsonl \
  --limit 32 --seed 310 --max-tokens 512 --out /work/c8-mtp1-gsm32.jsonl
```

随后同硬件/权重/OPP/TP/eager/context/batch参数、同dataset SHA与indices运行FP16 KV对照；分别比较得分、逐题对错变化、解析缺失、截断和token使用量。MTP影响另以同缓存精度开关作对照；FP16/BF16或W8目标基线仍须在对应权重下独立评估。尚未开始/完成的任务不能写成验收通过。
