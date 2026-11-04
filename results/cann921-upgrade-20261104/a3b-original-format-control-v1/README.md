# a2原35B W8A8服务窄范围格式对照

18/18请求符合预期。诊断脚本与27B相同，输入除model字段外严格相等；对照汇总见相邻dense-quality-format-pair921-16k-v1/a3b-narrow-comparison.json。

只对恢复后的a2原服务发短请求，未重启/改服务、未访问a1。provenance.json保留脚本/生产manifest哈希、执行前健康和models.json，service-health-after.json确认完成后仍HTTP200。与27B的模型、运行栈和上下文配置不同，不能作基础模型能力、量化或单kernel因果结论。该18条不等价于会议业务能力评估或正式性能。
