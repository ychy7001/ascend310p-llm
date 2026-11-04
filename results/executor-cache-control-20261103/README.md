# ACLNN_CACHE_LIMIT 原始GDN对照

同镜像、原始生产算子，默认配置及ACLNN_CACHE_LIMIT=0各48次独立递推对照：21/48、25/48在门限内；两者完整Torch-WY调用链各12/24越界，frozen-WY-clone本轮各0/24。没有消除异常，不选为GDN修复。发生率差异不作为性能或显著性证据。

两个测试子进程返回0并生成原始指标，但控制器最终status=failed，因为恢复a2临时服务时触发systemd启动频率限制。该清理失败保留不改写；之后已reset-failed并交接Dense独立启动screen，结束再恢复。不可把控制器失败解释为测试未运行，也不能把子进程返回0解释为数值通过。
