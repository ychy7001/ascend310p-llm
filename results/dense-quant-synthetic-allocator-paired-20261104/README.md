# 不含模型权重的随机INT8/零输入507015复现

生产固定镜像，原始生产布局[M4096,K8704,N5120]；随机INT8权重[-128,127]，channel-scale均匀正数，activation与pertoken-scale均零。每次独立进程，False/True交替，seed310～314。完整输出必须finite且全零。

False4PASS/1FAIL507015；True2PASS/2FAIL507015，最后1项未输出tensor准备标记，设备初始化等待超过180s后清理，总185.27s，不计Matmul故障。原始status=failed（最后超时）保留，不能用全局状态代替每项裁决。

3次失败均原QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21/hash5247287448945562503；不是模型权重特例，False不是修复。真正进入算子的9项结果、命令、日志、实际代码SHA均归档。

可共享复现包results/upstream-dense-quant-repro-20261104.tar.gz，仅包含生成数据代码、结果JSON和README，无权重或业务内容。未发布上游，未宣称已修复。
