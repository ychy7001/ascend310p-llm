# 真实 Qwen3.6 LM head W8A8 独立实验

使用a2芯片1、真实 `lm_head.weight`（BF16，[248320,2048]）。310P `QuantMatmulWeightNz` 对单次输出N有65535上限，原始整词表调用明确失败：`x2 last dim ... larger than 65535`。因此实验按4个词表分片，每片62080输出，逐片执行动态A8＋W8 NZ MatMul。没有改服务。

| batch | W8A8四分片耗时 | 相对L2误差（对ND FP16参考） |
|---:|---:|---:|
|1|3.225ms|1.28%|
|2|3.223ms|1.37%|
|3|3.222ms|1.40%|
|8|3.236ms|1.32%|

权重从FP16/BF16存储约1.017GB降至W8约508.6MB，另有约0.99MB per-channel scale；约50%权重存储节省。输出有限，误差可测。

**不能把原始记录中的127倍写成W8A8加速。** FP16参考使用ND `torch.matmul`，单次约412ms，只是数值参考，不是生产的NZ LM-head实现；尝试对FP16分片执行`npu_format_cast(..., FRACTAL_NZ)`在本CANN/310P实验容器中失败，日志报TransData unsupported/groups=1。因此当前没有公平的FP16-NZ分片基线，也不能宣称W8A8端到端加速。W8A8 3.22ms本身说明算子可运行，是否优于vLLM生产LM head需在框架真实权重加载路径补齐NZ基线后再判定。

量化误差约1.3%相对L2，尚未做生成质量/top-k一致性；不能直接接入默认服务。下一步若继续：复用vLLM实际LM head权重加载的NZ布局，做同形状FP16/W8A8单token logits top-k和Graph测试；若无可用FP16-NZ分片接口，先向CANN/华为确认N>65535和FP16 NZ转换限制。
