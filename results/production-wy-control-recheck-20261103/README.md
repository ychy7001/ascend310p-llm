# WY 配对复核（2026-11-03）

控制器 recheck-native-wy-control.py，远端 production-wy-control-recheck。A/B/A′ 保持同生产镜像、TP2 W8、Graph[1,2]、MTP0与262144上下文，a1未修改。A′四档Prefill对A漂移约−0.88%/+0.15%/+0.34%/+0.36%；候选对A′仍约+39.65%/+53.79%/+45.56%/+39.03%。正式请求每档3次，不能作为置信区间。

isolated-order* 是NPU空闲后原路径/候选各12次交替诊断。原路径3/12在严格门限内、候选12/12；两路径最终状态均逐位稳定，原路径偏差集中token56。快照保留远端native-gdn/isolated-order；本地保留指标和CPU快照分析，不下结论具体内核根因。

原配置A′在三并发整套会议4出现同样负责人轮换错配；candidate/control-meeting-owners.jsonl是两配置各6次串行复测，负责人映射均正确。不能用串行通过验证并发语义，也不能把首个候选错误归为候选独有回归。business-quality.jsonl自动28项只检查完成/工具/schema，须另做语义审阅。

A′完整长回放26/26、508.42s，对照最初A的508.40s，漂移极小；B为492.81s。控制器与最终清理均已完成，status.json和回放已归档。后续 batching/Profile 和算子诊断由独立控制器记录。
