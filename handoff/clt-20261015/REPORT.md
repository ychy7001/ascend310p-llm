# clt 部署验证与参数选择报告

2026-10-15（北京时间）。交付目标：让后端同事直接在clt复现当前单流长上下文服务。操作入口见[DEPLOYMENT.md](DEPLOYMENT.md)。

## 交付结论

默认采用TP1 / W4A8 / C16 / MTP1 / Decode Graph，112K档、batch2048、单请求、同步调度、Prefix off。clt为x86_64/PCIe EP/单颗310P1，可见44215MB设备内存。模型Qwen3.6-35B-A3B，原W4混合checkpoint包含MTP浮点权重，未在本轮重新量化或覆盖。

C16与C8都完成112000输入+128输出检索，但同容量核算C8净省约0.99GiB，112K实测Decode下降17.8%–23.1%。48G设备能承载目标上下文，因此优先保留C16的速度；C8作为容量受限备选。该结论不外推至其他并发、更大上下文或32G规格。

## 同条件性能

W4权重、TP1/MTP1/Graph/GDN/MindIEQ相同；max_model_len114688、util.90、batch2048、max_num_seqs1、prefix off、sync、固定128token输出。8K/32K为一次预热后三轮独立prompt的中位数；112K两位置各一次，没有同长度单独warmup。

| 输入token | C16 Prefill tok/s | C8 Prefill tok/s | C16 Decode tok/s | C8 Decode tok/s | C8 Decode下降 |
|---|---:|---:|---:|---:|---:|
| 8192 | 589.54 | 586.84 | 30.11 | 28.72 | 4.6% |
| 32768 | 527.05 | 523.34 | 24.35 | 21.28 | 12.6% |
| 112000，10%位置检索 | 395.76 | 392.62 | 13.13 | 10.78 | 17.8% |
| 112000，90%位置检索 | 397.69 | 394.23 | 13.82 | 10.63 | 23.1% |

Prefill各档差异不到1%；112K冷输入TTFT约4.7分钟，业务需设计合适的等待体验与超时。Decode为127/服务端Decode阶段时间，不是128/HTTP全程时间。

C16默认切换后再次完成五类sanity，8K预热+三轮中位数Prefill588.83、Decode30.20tok/s；容器RestartCount0、health200。复用前一轮相同配置长检索结果，切换后没有再跑一遍完整长测。C16与C8全对照没有掺入a1 TP2 W8A8历史吞吐，不能据此宣称W4相对W8的加速比。

## 正确性与上下文边界

- C16、C8均通过112000输入的前/后位置检索；之后9/9固定题通过，包括A/B/A复访。
- 8K/32K主对照两档MTP接受率均100%；112K early分别C16 70.7%、C8 81.4%，late均78.9%。合成重复输入接受率不能外推真实会议文本。
- 长检索强制ignore_eos输出128token，门禁只验证首段答案；之后重复模板不作为自然生成质量通过。五类sanity也不是完整精度评测。
- max_model_len114688限制输入+输出总长，已完成112000+128；C8曾验证114688输入+128输出正确HTTP400拒绝。不是硬件最大容量测试，通用241277-token日志不代表已验收上下文。
- 本轮未验收并发变长batch、JSON Schema、视觉、Prefix cache正确性、持续业务稳定性或完整精度集。

## 内存与峰值口径

| 指标 | C16 | C8 | 解释 |
|---|---:|---:|---|
| 长请求后设备占用 | 40560–40561MB | 35514MB | 总量44215MB，阶段结束快照 |
| 已采样最高占用 | 40561MB | 35518MB | 10秒采样；C16含启动，C8长测不含首次启动，可能漏瞬态峰值 |
| 以上最高值对应余量 | 3654MB | 8697MB | 当前自动配置，预留页数不同 |
| 实际缓存池 | 10.29GiB，223页 | 5.15GiB，148页 | C8另有224MiB共享转换工作区 |
| 相同107页缓存+工作区核算 | 约4.94GiB | 约3.94GiB | C8净省0.99GiB；尚未以此最小页数启动验收 |

当前启动就预分配112K档缓存，业务输入由8K增长到112K不会线性消耗额外几GB可用空间，主要变化是计算代价。按业务最大长度收紧缓存需要控制实际页数并重新验证，不能仅修改max_model_len或util便保证达到理论最小占用。

预算表见[CAPACITY-ESTIMATES.md](CAPACITY-ESTIMATES.md)。其中包含2GiB建议安全余量，但全部属于收紧缓存后的核算，不是当前服务实测峰值或正式部署保障。标称48G的可见容量只有44215MB，不能直接按48GiB预算。当前未证明32G可用。

## 适配方法概要

- W4权重通过现有打包和量化元数据加载，设备端激活量化、自定义310P矩阵路径支持Routed Experts；运行时适配与OPP/torch扩展共同构成实现，不能只换checkpoint。
- MindIEQ grouped native-layout路径用于达到token阈值的Prefill计算；小Decode保留已验证分支。当前阈值128。
- GDN PR17640相关扩展替代早期CPU WY路径；clt同步元数据适配支持当前同步调度。未将独立GDN ragged-MTP上游PR的并发验证结果自动当作本部署验收。
- C8备选为INT8存储→有效历史KV反量化为共享FP16/NZ→原生Attention，尚非原生INT8 KV Attention；长历史转换和额外读写解释了Decode开销增长。当前默认关闭C8，但保留兼容启动器和库。
- Graph覆盖Decode，保留MTP1；本轮没有证明MTP2优于MTP1，也未增加并发或开启Prefix以混入额外收益。

## 复现记录与后续优先级

公开仓库：https://github.com/ychy7001/ascend310p-qwen36

权重：https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P

原始数据在clt的`/home/clt/qwen36-kv-comparison-20261015/c16/logs/`及GitHub的`results/clt-kv-comparison-20261015/`；默认切换结果在`results/clt-context-selection-20261015/`。具体路径与命令见部署指南。

后续先用真实会议文本做质量与长请求压测，固定输入输出条件，再探索C16缓存预留收紧。Prefix Cache与并发单独验证，避免把冷热请求混为同一性能基线。MTP2、C8直接Attention和更大极限上下文不纳入当前默认交付承诺。

硬件已有PMBus Warning，尚未处理；软件测试通过不代表硬件健康告警已消失。此交付不包含驱动升级、完整系统重建或从空白服务器自动安装。
