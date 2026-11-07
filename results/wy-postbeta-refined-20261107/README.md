# raw-K Gram + Neumaier + 全128维refined求解（2026-11-07）

新kernel SHA `1e4b68e09ccae455c2c80415ee4ae50ae906aae37ff8ad05d048af208523009b`，父Neumaier候选SHA `e53600af2172159338673fc15c153fd5b40224265048ab41108af9de4804503e`。唯一新增变化是让全部K/V=128任务使用既有ProcessRefinedTask，而非按原gate选择fast doubling。原gate常量、阈值、scalar retry、生产kernel和worker均未改；不是cutoff搜索或阈值放宽。

隔离构建/安装422.97秒，52个制品；设备G0 76.42秒、ExitCode 0、cleanup_passed=true；原CPU FP32 oracle独立裁决20.91秒、ExitCode 0。窗口后服务inactive、rootful/rootless容器为空、NPU无进程。

| 固定集合 | 候选通过 | Torch通过 | 候选W/U超差元素 |
| --- | --- | --- | --- |
| 120快照 | 120/120 | 98/120 | 0/0 |
| 12触发 | 12/12 | 0/12 | 0/0 |
| 480边界 | 480/480 | 403/480 | 0/0 |

全部五项在三集合的no-worse均通过，g与原CPU参考逐元素一致、误差平方和0。132快照/触发的串行/排队和480边界的重复检查均通过，两卡合成30/30通过，进程加载期望新库。严格G0 passed=true。

与父候选相比，唯一新增的全128 refined dispatch消除了剩余4个U超差元素，支持快速求解/中间舍入路径是这些残差来源；尚未证明其中某一个乘法是唯一根因。

**这些不是独立盲测样本。** 480边界来自两个已知失败位置×30个WY层×若干chunk，候选设计参考了这些数据。历史上G0通过不能预测G1，因此本次只支持进入预冻结的固定模型筛查，不构成模型精度、OCR质量、性能或生产采用通过。全refined可能付出速度代价，尚未测量。

下一步原A/A′/B固定五点筛查：replay-03 p2结构性点精确匹配R；replay-00 p2/replay-06 p0/meeting-16384-0 p0仅记录；replay-14 p5目标截尾单列不可判定。新selected-policy在模型取数前写入，既有协议/规则/R/worker SHA和历史G1失败结论保持；不自动进入G2/性能。

归档runner放在runtime对应记录目录、tools同级、base为runtime根。构建依赖既有固定源/镜像；完整原始设备输出保留私有临时证据。该实验制品不是可部署生产镜像。
