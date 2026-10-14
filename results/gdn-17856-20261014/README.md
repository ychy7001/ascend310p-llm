# 310P GDN + MTP Issue #17856 实机记录

完整协议、修复说明和复现命令见 [复现报告](../../docs/gdn-17856-reproduction.md)。

- `operator-original.log`：旧版算子的确定性异常复现，包含状态预取对照。
- `eager-v2.log`：修复版直接算子 Eager，42/42。
- `graph-v2.log`：修复版直接算子 Graph，42/42。
- `wrapper-graph-v2.log`：修复版 Python wrapper + Graph，42/42。
- `opp-order-check.log`：框架初始化改变 OPP 优先级时失败，固定优先级后 21/21。
- `http-baseline.log`：原单请求服务排队处理 16/16 短文本和 JSON Schema。
- `http-c8-concurrency-failure.log`：C8 四请求候选触发既有单请求限制，失败记录。
- `http-c16-candidate.log`：修复版 C16 四请求候选，16/16 通过。
- `model-candidate.log`：完整候选启动记录，含前期失败、最终 Graph 捕获和模型回归。

日志中的最大误差来自 FP16 NPU 结果与 FP32 CPU 参考比较。同组数据在多条执行路径重放，不能将调用数当作独立随机样本数。

旧版主服务的短请求成功不代表未触发场景已安全；本补丁是否解释历史 MTP 接受率下降尚无证据。模型集成结果另见复现报告，不能用单算子测试代替模型验收。
