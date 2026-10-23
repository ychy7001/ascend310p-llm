# 112K Decode profile

E5参数、制品不变，114560输入/128输出，独立PROFILE_DIR。工具首块之后开启profile，避免Prefill混入。profile时间非benchmark；可能漏掉首个Decode步骤，解析按捕获kernel求和，不等于墙钟占比。上一轮8K结果在../decode-profile-20261014。

启动进程4021703，采集控制器4022011，a2目录/home/ychy/ascend-lab-runtime/mindieq-r3-20261013/decode-profile-112k-20261014。等待实际进程结束并检查state，不依据旧DONE判断。控制器EXIT自动恢复普通服务。工具analyze_decode_kernels.py支持gzip原始表及8K summary对照；长上下文结果现已导出，见summary.json及kernel_details.csv.gz。

复现使用../decode-profile-20261014/run.sh，设置PROFILE_OUTPUT_DIR=$work/decode-profile-112k-20261014 PROMPT_TOKENS=114560 OUTPUT_TOKENS=128；启动服务额外PROFILE_DIR指向同目录，其余沿用../mindieq-shared-gate-20261014/start-service.sh。先停止同芯片容器，再启动profiler服务，最后启动run.sh控制器。当前配置无需转换或下载权重。

只读检查已确认容器实际Qwen3_5主模型及MTP都创建ParallelLMHead(vocab_size,hidden_size,quant_config=self.quant_config)，compute_logits经LogitsProcessor调用；容器路径/vllm-workspace/vllm/vllm/model_executor/models/qwen3_5.py:477及qwen3_5_mtp.py:381。运行日志quantization=None；现有W4替换集中routed experts，LM head未量化。该证据支持8K trace的2048×248320浮点矩阵乘来自LM head，但此轮未添加模块栈标记，仍不对每条kernel做绝对归属。

补充源码证据：inspected-source-sha256.txt锁定实际容器版本。310P现有W8A8动态Linear使用npu_dynamic_quant+npu_quant_matmul及对称per-channel scale，process_weights_after_loading做NZ/transpose；源码包含激活动态量化精度问题注释，不能因此直接把LM head改为默认量化。若长trace证实其优化价值，下一阶段优先真实LM head小矩阵batch1/2/3测试与logits/top1误差，随后完整生成/MTP接受率门禁。

分析工具用8K原始167731条kernel重新计算总和3116531.404us，并自对照原summary全部30项比例/均值一致。当前112K请求已发出，尚处Prefill；后续不以此请求的profile耗时报告产品吞吐。

## 实际采集结果

114560输入+128输出完整通过，捕获167083条kernel，耗时求和7.237675秒。首块后采集可能遗漏初始步，stop时RECORD周期警告与8K一致；按捕获范围解释，不代表全请求墙钟比例，也不把profile请求当benchmark。

| 路径 | 8K占比 | 112K占比 | 8K单次平均ms | 112K单次平均ms |
|---|---:|---:|---:|---:|
|Paged Attention|6.55%|36.77%|0.415|5.544|
|C8转换|3.87%|21.78%|0.245|3.284|
|MatmulWeightNz|47.12%|20.23%|0.287|0.286|
|W4 Expert|15.70%|6.79%|0.149|0.150|

两类KV相关算子112K合计58.55%，长历史转换与attention成为优先优化对象。Attention与转换各480次（8K各492次），不能直接比较累计时间当同样调用次数；表中另列单次均值。原生PA112K在真实MTP2路径5.544ms不同于Q1微测试2.802ms，不能拿Q1当MTP2成本；已验证微测试不适合直接外推整模型。W4/普通Matmul平均基本不变，份额降低是KV成本上升。

工程次序：先保留缓存量化精度和Graph正确性，对转换做实际候选优化；再评估分块融合/直接C8 Attention（已有早期候选不代表性能通过）。LM head更适合短上下文Decode优化，可并行准备但不干扰基准。所有新路径须过真实生成/MTP接受率/长请求门禁，默认不变。

采集控制器已结束，state CAPTURE_DONE；无profiler恢复进程4025309已启动；独立恢复验证4022919随后terminal，restoration-state DONE，health通过、sanity5/5，原始结果restored-sanity.json。
