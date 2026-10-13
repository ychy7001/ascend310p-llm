# 质量与缓存回退验证协议

有限生成sanity不是完整精度验收。数据集留在实验目录，不进入Git；历史原始响应保留在实验机，后续逐题响应、usage、截断原因和关键日志统一归档到仓库results/。

## 已测请求复用小样本

experimentally verified：TP1/W4A8/C8/eager/MTP=0/4K，8个算术/排序/提取/中文题严格答案8/8；A/B/A中间3072输入+64输出干扰，前后答案文本一致、未泄漏标记。脚本`tools/check_cache_request_reuse.py`，原始`c8-reuse-quality.json`。不能推广为正式准确率或全部allocator隔离。

该脚本现在记录前后spec_decode Prometheus指标，供MTP候选检验实际draft/accept/reject覆盖；计数器变化与服务生成结果需一起看，不能把health或MTP配置存在当成有效verify证据。

## 跨页拒绝槽覆盖探针

`experimental/check_c8_rejected_slots_310.py --library <libqwen36_c8.so>`对prefix63/64/127写入两枚错误draft的K/V，再限制有效长度读取，要求与写入前完全一致；覆盖下一提交位置，和新建干净C8缓存的原生PA对照。测试包括scale覆盖和页边界。experimentally verified：原版和融合write库均在真实310P3通过上述3个边界，max_abs_vs_fresh_c8=0；只证明存储/PA语义，不替代scheduler及linear state rollback。

## GSM8K固定子集协议

数据来自[官方grade-school-math](https://github.com/openai/grade-school-math)，固定commit `3101c7d5072418e28b9008a6636bde82a006892c`、文件`grade_school_math/data/test.jsonl`。实际1319行，SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`；经GitHub API下载，不重新分发数据。

首轮32题、random seed310、选中indices排序后顺序执行；temperature0、enable_thinking=false、max_tokens512、固定zero-shot后缀，严格提取最后的`####`数字。保存全部响应，未解析或截断不自动当正确，不用最后一个随意数字兜底。此协议不是标准lm-eval完整GSM8K分数；32题只适合早期回归定位，不足以证明小幅准确率变化。

```bash
python3 tools/eval_gsm8k_http.py --base-url http://127.0.0.1:18016 \
  --model qwen36-w4a8-c8 --label c8-mtp1 --dataset /work/gsm8k-test.jsonl \
  --limit 32 --seed 310 --max-tokens 512 --out /work/c8-mtp1-gsm32.jsonl
```

随后同硬件/权重/OPP/TP/eager/context/batch参数、同dataset SHA与indices运行FP16 KV对照；分别比较得分、逐题对错变化、解析缺失、截断和token使用量。MTP影响另以同缓存精度开关作对照；FP16/BF16或W8目标基线仍须在对应权重下独立评估。尚未开始/完成的任务不能写成验收通过。

## 第一轮GSM子集结果

experimentally verified：TP1/W4A8/C8/eager/4K/.90、MTP=1、max-batch256、max-seqs1、无prefix cache，固定32题32/32、parse_missing=0、truncated=0；6875输出tokens。GSM期间draft3496、accepted3365、reject131；之后8题/A-B-A复用仍通过。原始c8-mtp1-gsm32.jsonl与metrics留a2。FP16 KV+MTP同配置、同seed/indices对照正在加载与排队；尚不能用单侧32/32证明精度等价，也不作为完整GSM或全面质量分数。

## 匹配FP16 KV对照已完成

experimentally verified：相同checkpoint/OPP/extension、TP1/eager/4K/.90/MTP=1、max-seqs1/max-batch256、无prefix cache、同SHA/indices/prompt/512输出上限；FP16 KV为31/32，parse_missing=1/truncated=1，C8非融合独立层为32/32。唯一差异index205：FP16正确推导过程中512tokens处截断，未出现####98；C8在上限内回答98。其余31题数值答案相同，全文仅9题完全相同。这个差异不能解释为量化提高准确率，也不能从32题证明精度等价。脚本compare_gsm8k_runs.py拒绝未完成或协议不同的runs；原始fp16kv-mtp1-gsm32.jsonl。FP16的8题/A-B-A也通过。

## 32K配置MTP2实测小样本

experimentally verified：TP1/W4A8/FP16KV/eager/32K/.90/1024:512/maxseq1/prefix off，MTP1与MTP2各自toy严格8/8，A/B/A文本一致、无标记泄漏；两组32256中间needle均通过。MTP2的p128吞吐窗口含warmup，draft tokens620/accepted490，实际第二位置接受229且存在130枚未接受草稿。数据支持两枚草稿实际参与verify，不等于所有拒绝情形或长上下文精度已验收。两组都完成固定128/8192/32256输入、输出256的三轮有效usage请求。C8与Graph仍仅验证一枚草稿组合；新C8有效长度Graph集成的固定GSM32对照已排队，原始题目/输出不提交。

## 32K C8/Graph匹配质量对照：eager组完成

experimentally verified：TP1/W4A8/C8 fused write/split_by_type/MTP1/eager/32K/.90/1024:512/maxseq1/prefix off，固定同GSM数据SHA、32indices/seed310/temperature0/enable_thinking=false/max_tokens512，32/32正确，parse_missing=0，truncated=0。Graph全范围和有效长度早期单侧GSM记录仍按历史实验保留；最新 C8+Graph 4K/32K/112K 已通过 sanity、toy、A/B/A 和 needle，但不把小样本宣称为全面质量等价。原始eager文件仍留实验目录。
数学窗口draft3544/accepted3413，实际未接受131枚草稿，说明本次回归覆盖了真实拒绝；仍不是所有rollback情形的证明。

32K Graph全范围组已真实capture/replay并开始匹配GSM32。index205暂缺严格答案：512token截断，predicted=None；原始推导已得到x=98并校验均分93，但在讨论另一分支时耗尽预算。eager同题510token完成####98。此差异仍按协议记错，不把中间98当最终答案，不声称Graph质量等价，也不能仅据截断认定计算错误。完整32题结果和新有效长度组尚未完成；后续可将同题两侧输出预算提高做定位，但不得覆盖现有512token原始协议。

32K Graph全范围组匹配GSM32完成：31/32，parse_missing=1/truncated=1；唯一严格得分差异仍index205的512token截断，推导中得到98但无####最终答案。toy8/8与A/B/A复用通过（文本一致、无泄漏）。尚未提高输出预算重测该题或完成有效长度组，不声称Graph质量等价。

32K Graph设备有效长度组完成同协议GSM32：31/32、parse_missing=1/truncated=1，唯一严格差异仍index205截断。与Graph全范围组逐题numeric答案无变化、12题全文一致；toy8/8与A/B/A文本一致、无泄漏。该结果没有发现新增小样本回归，但不是数值完全等价或完整精度验收。尚未提高该题输出预算定位，FP16 KV Graph对照也未完成。

## N128 / 112K / C8 / MTP2 / Graph 子集回归

2026-10-13完成相同数据集SHA、seed310、32题indices、temperature0、thinking关闭、512token上限的回归：31/32，parse_missing=1、truncated=1；唯一未得分题index205在512tokens处截断。该结果不证明与MTP1精度等价，也不是全面GSM8K成绩。完整响应和服务日志见`results/n128-quality-20261013/`。首次客户端执行只留下1题，没有summary，未计入完整结果；另以新文件完成32题，保留了首次文件而没有覆盖。
