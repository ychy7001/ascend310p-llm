# Dense ACLNN_CACHE_LIMIT=0 启动screen

唯一主要变量为ACLNN_CACHE_LIMIT=0；保留原始Graph[1,2]、生产镜像、TP2、262144上下文及原始算子。启动仍失败507015，故障kernel仍为QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21（aclnnQuantMatmulWeightNz）。没有进入HTTP请求，无吞吐/质量结果，候选不成立。

GDN缓存对照后的unit恢复曾触发systemd启动限制，not-started目录保留该守卫停止。reset-failed后设备空闲确认，本轮启动/清理正常完成；最终临时unit active及健康探测单独见continuation-final-state.json，不由active推断模型健康。
