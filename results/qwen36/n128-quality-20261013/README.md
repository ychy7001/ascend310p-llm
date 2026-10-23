# N128 / C8 / MTP2 / Graph / 112K GSM固定子集

2026-10-13，TP1/a2第一颗310P3；服务参数与
`results/qwen36/n128-product-20261013`一致，完整启动日志在本目录。

相同GSM dataset SHA、seed310、32indices，temperature0、thinking关闭、
512输出上限：31/32、parse_missing=1、truncated=1；index205达到512tokens
上限，未输出可解析最终答案。样本和全部响应见gsm32.jsonl。不将此解释
为全面精度或与其他配置统计等价。

最初客户端只有1题记录且没有summary，未计入；之后使用新文件完成评测。
metrics-after.txt为评测完成后的快照，但没有可靠的本次对应before，不能
据此计算该子集的MTP接受率增量。后续匹配MTP1必须同时保存前后快照。
