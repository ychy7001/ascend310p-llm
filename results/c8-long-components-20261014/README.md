# 长上下文C8转换与Attention拆分

使用a2芯片1、310P3/CANN9.1/torch-npu2.10，沿用现有device-length C8库，未修改服务或算子。芯片0的112K基准完成后才执行，避免干扰。run-micro.sh记录容器命令；实际代码tools/bench_c8_long_components_310.py。

单层Q1、16 query heads/2 KV heads/head dim256，synthetic INT8缓存+FP16 scale、identity block table；对照为同C8码重建FP16缓存，不是原浮点模型精度。缓存NZ内容和Attention输出逐位一致（三种长度）。每项5次eager预热，Graph捕获后5次replay预热，再20次交替测量，记录host replay+synchronize时间，包含主机调度开销，非纯设备kernel事件计时。

| 历史token | C8转换ms | 原生PA ms | 转换+PA ms | C8存储MiB | FP16 scratch MiB |
|---|---:|---:|---:|---:|---:|
|8192|0.316|0.312|0.561|8.5|16|
|32768|1.013|0.871|1.841|34|64|
|114688|3.340|2.802|6.028|119|224|

112K组合约为native PA的2.15倍，C8转换耗时不可忽略；不能外推整模型加速比，也未覆盖MTP多查询、真实路由或实际block table。现有C8省常驻缓存，但依然需要共享FP16 NZ scratch并读写长历史；下一步用真实Decode profile核对占比，再决定优化转换还是尝试分块融合Attention。原始20次样本及日志随附。
