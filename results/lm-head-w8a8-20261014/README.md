# 真实 Qwen3.6 LM head W8A8 独立实验

使用a2芯片1、真实 `lm_head.weight`（BF16，[248320,2048]）。310P `QuantMatmulWeightNz` 对单次输出N有65535上限，原始整词表调用明确失败：`x2 last dim ... larger than 65535`。因此实验按4个词表分片，每片62080输出，逐片执行动态A8＋W8 NZ MatMul。没有改服务。

| batch | W8A8四分片耗时 | 相对L2误差（对ND FP16参考） |
|---:|---:|---:|
|1|3.249ms（Graph 2.998ms）|1.40%|
|2|3.444ms|1.14%|
|3|3.479ms|1.21%|
|8|3.464ms|1.34%|

权重从FP16/BF16存储约1.017GB降至W8约508.6MB，另有约0.99MB per-channel scale；约50%权重存储节省。输出有限，误差可测。本次 batch=1 top-k overlap 为 top-1/5/10/50 均 100%；此前独立重复曾出现 top-5 80%、top-10 90%、top-50 96%，说明随机 hidden state 下指标有波动，必须以多种 hidden states/真实 logits 校准后再判断。这只是与同一 ND FP16 分片参考的算子输出比较，不代表完整生成质量。

**不能把原始记录中的127倍写成W8A8加速。** FP16参考使用ND `torch.matmul`，单次约412ms，只是数值参考，不是生产的NZ LM-head实现；尝试对FP16分片执行`npu_format_cast(..., FRACTAL_NZ)`在本CANN/310P实验容器中失败，日志报TransData unsupported/groups=1。因此当前没有公平的FP16-NZ分片基线，也不能宣称W8A8端到端加速。W8A8 3.22ms本身说明算子可运行，是否优于vLLM生产LM head需在框架真实权重加载路径补齐NZ基线后再判定。

量化误差约1.1–1.4%相对L2。本次改为预分配输出并对四个分片做 `copy_`，batch=1 的 Graph 捕获通过，Graph 输出相对 eager 的最大绝对误差为0，耗时约2.998ms，较 eager 3.249ms 下降约7.7%；仍有 storage_offset 警告，且尚未做完整生成质量门禁，不能直接接入默认服务。下一步若继续：复用vLLM实际LM head权重加载的NZ布局，做同形状FP16/W8A8单token logits top-k和Graph测试；若无可用FP16-NZ分片接口，先向CANN/华为确认N>65535和FP16 NZ转换限制。
