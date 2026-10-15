# RC MTP2 Graph：draft 回传分段定位

此为诊断实验，不是性能优化或可发布修复。

- 基于`qwen36-mtp2-snapshot-draft-eager-20261015`创建独立容器
  `qwen36-mtp2-stream-stages-20261015`，保留accepted-token快照补丁。
- TP1 / C16 / W4A8 / MTP2 / target Graph / draft eager / 同步调度，
  max32768、KV3GiB、batch2048、专家chunk512、GDN和MindIEQ不变。
- 只增加`model_runner_v1.py`只读挂载，行为差异见`diagnostic.patch`，
  源文件与运行时校验见`source.sha256`和`runtime.sha256`。
- 默认服务先停止保留，候选单独占用NPU。完成诊断后恢复原MTP1并重新验收。

分段标记及含义：

| 标记 | 操作 |
|---|---|
| producer_sync_begin/done | 当前生产stream执行显式同步 |
| wait_begin/done | copy stream wait生产stream后显式同步 |
| copy_begin/copy_enqueued | 向pinned CPU buffer提交non-blocking copy |
| event_sync_begin/done | copy后record event并同步 |

屏障会改变原始时序。若只输出producer_sync_begin，只能定位到copy前生产stream
已有未完成工作，不能确定具体kernel；若全部通过，说明串行化会改变故障表现，
仍需单独减少屏障并做质量/性能对照。任何情况下均不删除原有event同步。

短质量检查使用现有`check_cache_request_reuse-fixed.py`，外层等待启动900秒，
health通过后命令限制180秒；失败不做吞吐测试。所有记录保留到此目录。

## 本轮结果

候选完成启动和Graph捕获，health200。首个27-token短题返回HTTP500，完成题数0。
仅输出`producer_sync_begin`，没有`producer_sync_done`，也未进入wait/copy阶段。
显式生产stream同步报告`507018`、AI CPU执行异常，kernel为`FloorDiv`，
`stream_id=34/task_id=21701/PID=73083/errorCode=0x2a`。
服务日志`service.log.gz`、摘要`failure-excerpt.txt`和设备TS摘录
`device-task-excerpt.txt`相互对应。

因此，本轮异常发生于draft回传copy提交之前，原先event等待不能作为D2H或event
本身故障的证明。诊断屏障会改变执行时序，不能据此宣称历史所有停滞均已完全归因。

源码候选位置：`Ascend...Proposer.attn_update_stack_num_spec_norm`在第二个及以后
draft步骤使用`clamped_positions // block_size`计算slot，再使用`% block_size`
计算偏移，见`proposer-slot-source-excerpt.txt`。MTP1没有第二个draft更新，
因此这是与MTP2特有故障相符的优先检查点；尚无op级调用栈证明本轮失败一定出自此行。
下一步在此处做前后同步/输入dtype和block_size观测，或独立验证有界位置整数的
替代实现，再用只改该算术的候选做对照。不能全局替换Tensor整数除法。

本轮未执行吞吐或长上下文测试，未把诊断屏障合入默认源码。

原`qwen36-orangepi`已恢复：health200，短题8/8，A/B/A前后文本一致，无canary
泄漏，见`restored-mtp1-quality.json`。候选停止保留，只有原稳定容器运行。
