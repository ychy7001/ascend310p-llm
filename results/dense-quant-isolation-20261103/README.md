# Dense 动态量化矩阵乘合成隔离

每个形状/布局独立进程，生产镜像 torch-npu.post2/CANN9.1-beta，M=1/4096，(K,N)=(5120,17408)/(8704,5120)，采样3行×32列与CPU INT32矩阵积及显式scale相乘对照。仅是算子正确性诊断，不是性能测试。

生产布局 `format_cast(weight[N,K],NZ).T` 四项均PASS；`format_cast(weight.T.contiguous(),NZ)` 两项均数值FAIL；ND两项均被API拒绝。不得将替代布局作为修复。

原始模型仍在 MLP down_proj 启动warmup失败，随机合成张量未复现。形状通过不能证明模型真实激活/stride/format/scale或调用历史无误，下一步捕获实际输入。完整命令、JSON和日志保留。
