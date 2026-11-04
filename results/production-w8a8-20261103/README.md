# 生产W8A8 TP2副本基线

会话日期2026-11-03，实验仅在a2的18106独立端点运行；a1只读审计。a1与a2镜像digest、W8权重、生产推理参数一致，驱动分别为26.0.rc1、26.1.1；不能把a2结果称为a1在线实测。

- 启动来源：../production-a1-audit-20261103/manifest.filtered.json，以及仓库benchmarks/ascend310p/production-manifest.json。
- C1：1K/4K/16K/32K输入、输出256，seed0、temperature0、ignore_eos，warmup1+正式3。summary-c1.json汇总；12/12有效。
- 并发：1K/256，C1/3/4各3轮，24/24请求成功。summary-concurrency.json；只有三轮，不能评价可靠P95。
- 业务形状回放：26/26，76631输入/20690输出，508.4s，冷缓存命中增量0。使用固定合成token输入、阶段依赖和输出长度；不是私有业务文本或纪要质量评分。
- 业务质量：28条完成/工具/schema自动检查通过，纪要另有已知语义风险，见business-semantic-review.json；不声称语义28/28。
- 事实门禁：原始14/16有两条“负责”前缀误报。原始日志、失败状态和重算16/16均保留；evaluation-correction.json解释评测器修复边界。
- GSM：固定64题zero-shot/greedy/thinking off/512输出上限，63/64；1条截断无最终答案，索引205。不是标准全量或few-shot分数。
- Profile：4K Prefill输出1、32K首输出后Decode，双rank共4份CSV及摘要；profile-artifacts.json映射范围/rank/文件。累计耗时占比不等于墙钟占比。
- 自动解析被torch-npu daemon限制拒绝；原始采集成功后，同镜像/CANN离线analyse并补齐设备/驱动访问，最终四份CSV均成功。离线无设备尝试的msprof错误和全部命令/日志保留，status-profile-export-pending.json为恢复前状态。
- 资源：resource-samples*.jsonl及resource-summary-initial.json；约5秒间隔的npu-smi设备采样峰值，不是精确HBM/allocator峰值。

事实评测器误报后实验服务重启一次，只补跑回放/GSM/Profile，先前有效性能未重跑。原始普通计时与诊断采集分离；服务日志区分service-before-evaluator-restart.log.gz和service-final.log。采集结束后服务已停止，没有改动a1生产配置。
