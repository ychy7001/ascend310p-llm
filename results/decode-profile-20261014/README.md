# 真实MTP2 Graph Decode采集（8K）

E5同参数同制品，仅增加PROFILE_DIR开启现有torch profiler。工具tools/profile_http_decode.py在首个流式文本后调用start_profile，128输出结束后stop_profile；可能遗漏最初Decode步，无法作为性能基准或完整请求trace。未包含请求warmup；启动已进行Graph捕获。停止时提示RECORD状态可能产生不完整数据，因此只分析实际捕获范围，不推算全请求调用次数。

p8192-o128.jsonl校验8192输入/128输出、采集启停成功。kernel_details.csv.gz为原始导出（gzip -n），167731条kernel，耗时求和3.116531秒；不同stream可能重叠，以下占比不是墙钟占比。op_statistic.csv及形状分组保留。

| 算子 | 累计ms | kernel耗时求和占比 |
|---|---:|---:|
| MatmulWeightNz |1468.646|47.12%|
| QwenW4A8Int4MatmulV310 |489.242|15.70%|
| Paged Attention |204.225|6.55%|
| Matmul TransData |185.771|5.96%|
| RecurrentGatedDeltaRule |121.251|3.89%|
| C8转换 |120.693|3.87%|

MatmulWeightNz形状分组显示最大一项123次、708.216ms，权重NZ[128,15520,16,16]，对应2048×248320，与config的hidden_size2048/vocab_size248320吻合，推断为浮点LM head，需源码或标注再次确认。单这一项约占kernel耗时求和22.72%。其次[128,768,16,16]约396.379ms；不只专家W4或C8值得优化。形状不是模块调用栈，不能把未标注项直接定性。

结论：真实Graph内部采集有效。8K的C8转换仅约3.9%，不应据单层112K测试把全部Decode成本归因C8。下一步采集长上下文对照，核对LM head与其他浮点投影来源，再用独立候选验证量化/计算优化和精度。当前未改动任何生产计算路径，未将候选设为默认。

run.sh记录采集控制与EXIT恢复；start-service由results/mindieq-shared-gate-20261014/start-service.sh复制到远端start-shared-profile-base.sh。启动命令：PROFILE_DIR=$work/decode-profile-20261014 nohup bash $work/start-shared-profile-base.sh > $work/decode-profile-20261014/start.txt 2>&1，然后nohup bash $work/run-decode-profile.sh。原始完整trace保留a2同目录，未上传大体积trace JSON。采集控制器结束后已启动无profiler恢复进程4019591；提交时仍加载，不能声称恢复健康检查完成。
