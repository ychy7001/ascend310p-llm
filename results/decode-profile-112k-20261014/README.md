# 112K Decode profile（正在执行）

E5参数、制品不变，114560输入/128输出，独立PROFILE_DIR。工具首块之后开启profile，避免Prefill混入。profile时间非benchmark；可能漏掉首个Decode步骤，解析按捕获kernel求和，不等于墙钟占比。上一轮8K结果在../decode-profile-20261014。

启动进程4021703，采集控制器4022011，a2目录/home/ychy/ascend-lab-runtime/mindieq-r3-20261013/decode-profile-112k-20261014。等待实际进程结束并检查state，不依据旧DONE判断。控制器EXIT自动恢复普通服务。工具analyze_decode_kernels.py支持gzip原始表及8K summary对照；尚未生成长上下文结果。

复现使用../decode-profile-20261014/run.sh，设置PROFILE_OUTPUT_DIR=$work/decode-profile-112k-20261014 PROMPT_TOKENS=114560 OUTPUT_TOKENS=128；启动服务额外PROFILE_DIR指向同目录，其余沿用../mindieq-shared-gate-20261014/start-service.sh。先停止同芯片容器，再启动profiler服务，最后启动run.sh控制器。当前配置无需转换或下载权重。

只读检查已确认容器实际Qwen3_5主模型及MTP都创建ParallelLMHead(vocab_size,hidden_size,quant_config=self.quant_config)，compute_logits经LogitsProcessor调用；容器路径/vllm-workspace/vllm/vllm/model_executor/models/qwen3_5.py:477及qwen3_5_mtp.py:381。运行日志quantization=None；现有W4替换集中routed experts，LM head未量化。该证据支持8K trace的2048×248320浮点矩阵乘来自LM head，但此轮未添加模块栈标记，仍不对每条kernel做绝对归属。
