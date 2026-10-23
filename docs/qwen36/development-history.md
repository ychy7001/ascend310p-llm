# 从低比特可行性到112K产品候选的思路历程

本文按决策阶段梳理过程，不把后来的结果写成一开始已经知道的结论。完整历史由Git提交、docs/参数记录和results/原始数据追踪。

## 1. 起点：验证社区方法能否迁移，而不是预设INT4更快

初始参考是a1的TP2 W8A8部署与社区Qwen3.8-Flash-Next的310P实现。问题分为硬件路径、模型适配和实际收益三部分。先确认a2的310P3、CANN和镜像环境，隔离构建/服务，检查真实专家shape及社区源码，避免复制不同模型的checkpoint或整个fork。

没有官方310P W4部署方案，不等于底层所有INT4计算不可用。源码与实机执行确认int4b_t/Mmad的整数路径；A8拆成两个INT4 limbs组合计算。因此选定W4A8原型，但不把它宣传成单指令INT8×INT4，也不预设能超过W8A8。

## 2. 先打通真实权重、算子和模型加载

从Qwen3.6原BF16量化routed experts，采用ModelSlim SSZ per-channel INT4，避免从W8二次量化。G128只是scale重复存储和kernel ABI；不是独立group-wise标定。非专家层保留浮点，降低初次适配范围。

社区算子基础需要调整激活量化、元数据、真实矩阵和路由；Qwen3.6适配层还处理模型构造、TP所有权和checkpoint加载。使用独立Torch namespace和OPP避免覆盖官方扩展。先完成算子/短生成，再验证TP1/TP2，最后导出自包含权重、检查索引和哈希、发布ModelScope。

这一步解决“能正确加载并执行”，没有同时解决Prefill、准确率和产品稳定性。

## 3. 加入MTP：复用模型自身权重与现有框架

检查原Qwen3.6 checkpoint后确认自身MTP张量可用，无需借用3.5权重。曾有两个stacked expert张量漏列及MTP decoder误进入目标W4适配的问题，修正索引和前缀判断，保持draft浮点路径，并复用原有draft/verify机制。

先MTP1打通，后测MTP2。收益依赖接受率、输出长度与请求内容；最终同协议32K/256测试MTP2 Decode比MTP1快约24.8%，Prefill约降0.8%。这是有利负载下的方向性结果，不能推广为所有业务请求。

## 4. 从权重压缩转向总内存：研究动态C8

权重变小后，长上下文仍受KV、hybrid linear state、MTP与工作区制约。动态C8无需静态KV标定权重：设备端量化K/V为INT8、FP16 scales，读取时先转FP16 NZ复用原生PA。

早期独立buffer、padding和每层转换造成明显额外占用/成本。后续融合量化写入、去除无用尾部、按cache类型拆池、跨层复用scratch，并按物理预算与hybrid state约束限制admission。调度器报告的token容量与实际物理池不同，不能据旧310K估算宣称可运行310K单请求。

C8是容量路线，当前不是原生INT8 Attention全链路。32K匹配对照显示Prefill近似持平，Decode约下降5.5%～13.2%，但峰值少约2.5GB。因此保留C16速度对照，避免把C8无条件设成最快档。

## 5. 长上下文与Graph：从“启动成功”改为实际请求门禁

上下文逐档扩大，测试真实token usage、检索、cache复用及拒绝槽覆盖。过于乐观的容量报告促使我们增加hybrid物理预算检查。128K需要更激进内存利用率，结合产品100K以上目标，先选112K、利用率.95作为候选，而不是继续极限挤内存。

Graph初期在参数更新、capture中的CPU操作、动态长度及split graph方面失败。通过fake contracts、固定图形状、设备有效长度和mode0逐步打通。当前仍依赖updatable-graph诊断开关，已完成MTP1/MTP2及112K运行门禁；正式backend修复尚待收敛。

## 6. 从猜测瓶颈到profile与配对调优

a1 W8A8 TP2冷Prefill约1.3K是方向性参考，但TP、cache、调度和量化范围不同，不能除二就得出TP1上限。也不能拿TP2的AllReduce profile解释TP1。复核真实TP1 profile后，W4 MatMul累计约50.4%，优先处理专家算子、路由搬运与归约。

先扫描调度batch和expert chunk，拆开两者收益；大batch或640chunk并不自动改善长上下文。M16、M64候选没有形成可保留收益，继续保留M32。之后扩大Gate/Up Prefill N tile到128，Decode和Down策略保留；在eager和C8/MTP2/Graph两种组合中复核。

最新112K组合顺序配对：8K/128 Prefill364.49→391.59，+7.43%；32K/128为335.72→357.58，+6.51%；Decode差异小于0.3%。真实专家12个分组/尾块数值case通过。112K检索N64/N128均正确，但单样本总时间不能替代稳定性能评估。

## 7. 当前工程结论与下一步

目前得到可运行的TP1/48GB、W4A8+C8+MTP2+Decode Graph、112K性能候选。最新32题GSM固定子集31/32，一题截断；不能据此宣布全面精度通过。W4的明确价值是权重内存，N128的已观察收益是Prefill，MTP主要改善Decode，C8提供KV余量，prefix cache改善重复前缀TTFT；这些收益不能相加或相互替代。

下一步同时推进两件事：①把模型构造、C8 allocator和Graph monkeypatch收敛到明确可选后端，形成main上的小补丁；②用真实会议/问答、长输出、多位置检索与足量尾延迟验证选档。华为协助优先集中W4 Cube流水、原生C8读取接口和Graph生命周期。TP2 C8、32GB实卡和256K完整场景仍需独立实验。

代码模块与上游化边界见[代码入口](../code-map.md)，会议报告见huawei-adaptation-review-20261013.md（内部材料，已移出公开仓库），当前结果以[参数记录](parameter-selection-20261012.md)、[Prefill记录](prefill.md)及[质量记录](quality.md)为准。
