# Dense 模型内 down_proj 权重存储重建诊断

2026-11-04 UTC，生产固定镜像，TP2 W8A8_DYNAMIC，原始 GDN/FwdH/FwdO。同步 eager / Graph NONE，禁止用于性能比较。

仅为 K8704/N5120 down_proj 将权重转回物理 [N,K] ND、clone、转 NZ、转置；原始权重保留。每次 Matmul 前 CPU 逐元素严格相等。未修改激活、尺度或量化数学表达。

两个 rank 各加载 19.2416 GB。首个 down_proj 的权重重建均通过严格相等检查；shape=[8704,5120]、stride=[1,8704]、format29、offset0、storage44564480 bytes，与原权重一致。输入均为 ND [4096,8704]、offset0、storage35651584 bytes。

rank0 首次 Matmul 完成；rank1 同一次调用仍报507015，故障kernel QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21，hash5247287448945562503。服务未ready，无HTTP质量/性能结果。该重建方式不能选为修复，也不能据此排除其他张量存储或完整调用上下文。

status.json保留原始failed；控制器退出并恢复a2临时unit。tools/下保存实际运行的worker。前一轮独立重放6项实际仅覆盖rank0/device0，两个TP分片×两个物理设备的交叉复现另行记录。
