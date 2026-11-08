# W8 WY Decode 收益补验收

只在a2运行，a1不变。固定最终镜像bb0fa66f…，不挂载源码，使用内置入口；不改WY数学、CANN、状态修复和slot补丁。用于回答MTP0/1/2实际收益及并发下的代价。

`run_decode_ablation.py`顺序启动MTP0/1/2/2/1/0，每窗4K固定token输入、512token输出，c1/c4各两轮。每个并发档位64token预热，输入、seed相同、cache salt唯一，要求prefix hits为0。主Graph保持三档策略，实际捕获大小随验证宽度为[1,2,4]×(MTP+1)，避免将不合适的固定token档位偏置给某臂；其余max262144/seq4/batch2048/memory0.9/async1/prefix0一致，draft eager。两次启动及镜像/入口/实际CLI均记录；窗口之间清场并冷却60秒，持有a2重任务锁。

随后两窗`reference`复现仓库归档的a1旧镜像及原配置，只改实验host/port并映射同一份W8权重，不操作a1。旧配置batch4096/prefix1/capture[1,2]保持原样，使用唯一salt确保冷请求；该比较是整套配置比较，不能把全部收益归给MTP或WY，也不是a1实机测量。

每窗保留客户端逐请求TPOT/TTFT、端到端聚合吞吐、服务端请求阶段时间、MTP计数、原始输出及独立清场。`summarize.py`生成可提交汇总，原始text/events/cache/log留设备端或本机离线目录。

指标定义：

- **客户端TPOT**：(最终usage到达时间−首个非空文本时间)/(输出token数−1)。MTP可能一次发送多个首批token，故为长输出下的客户端近似，不声称token级精确时间。
- **客户端生成速度**：逐请求1/TPOT，报告中位数。
- **服务端Decode速度**：(输出token数−请求数)/请求Decode阶段耗时之和；并发时是按请求阶段累计时间加权的速度，不是聚合吞吐。
- **端到端聚合输出吞吐**：该轮总输出tokens/该轮wall time，包含Prefill和排队。

单一合成输入、每臂两次启动/每并发4轮，仅裁决当前配置与该负载的收益；不证明所有业务最优、长期稳定性或业务精度。若失败保留原始窗口并停止后续动作，不把已完成子项冒充整窗通过。
