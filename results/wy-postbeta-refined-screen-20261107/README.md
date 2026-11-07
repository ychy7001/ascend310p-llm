# 全refined WY固定五点模型筛查：不通过（2026-11-07）

候选kernel SHA `1e4b68e09ccae455c2c80415ee4ae50ae906aae37ff8ad05d048af208523009b`、manifest SHA `ad09658f727acd5ca52cab3391593ed0f6d909dfb1deb711fc17b454e2660f06`，此前严格固定612组G0已通过。新selected-policy在本轮任何模型请求前冻结，结构性点精确匹配R，近似平局只记录，截尾点不可判定；不回溯解释任何历史G1失败。

| 点 | 分组 | R | A | A′ | B |
| --- | --- | --- | --- | --- | --- |
| replay-00 p2 | 近似平局/仅记录 | 16 | 16 | 16 | 16 |
| replay-03 p2 | 结构性/必须匹配 | 97235 | 97235 | 97235 | **16** |
| replay-06 p0 | 近似平局/仅记录 | 15 | 15 | 15 | 15 |
| replay-14 p5 | 目标截尾/不可判定 | 104032 | 104032 | 104032 | 96069 |
| meeting-16384-0 p0 | 历史分组仅记录 | 16 | 16 | 16 | 97787 |

“近似平局”是预冻筛查分类，不宣称新B在该点的概率分布仍近似平局。meeting新B=97787如实保留；本轮否决仅由结构性点已足够成立，不需事后修改规则。

3000秒预算、实际646.16秒，控制器ExitCode 0，三个模型容器均ExitCode 0、无OOM，服务inactive、两类容器/NPU清空、cleanup_passed=true。独立分析从15份原始HTTP响应重新取top1，核对冻结context SHA和策略；双控制臂结构性点通过而B失败，eligible_for_full_G1=false。不启动完整G1/G2/性能，不采用。

B日志12条native调用记录验证期望候选opapi库。冻结worker输出LOCAL_RANK为null，独立核验采用日志的Worker_TP0/1和PID前缀辨认双rank；没有修改冻结worker。两rank均加载新库，无fallback。协议、规则、worker、R和生产kernel SHA未变，实验启动脚本仅指向新vendor，身份见freeze。

结论：消除局部CPU误差并未消除模型结构性偏差。该候选关闭采用线，不继续盲目调Gram/求解参数；G0不能替代模型验证。全refined性能未测，也不能沿用旧native的37–47%吞吐收益。完整原始日志、响应保留私有证据；归档只含结果/工具/哈希。

后续优先调查保持原Torch算术顺序的调度优化：63轮UT调用是否可用NPU图捕获/回放减少dispatch。先只读检查当前栈支持性，再决定短探针；不得据此改变任何冻结协议或直接采用。
