# Qwen3.8 Flash Next / a2双310P可行性调查

2026-10-18，只读调查；未停止3090/a2服务、未下载完整checkpoint、未执行模型推理适配。不构成“TP2已经跑通”的结论。

本机Strata IQ3_XXS已安装并完成此前小样本冒烟；调查时当前8080进程仍是已恢复的IQ2_XS。Strata的CUDA引擎、CPU专家、SSD PLE分层方案不是全权重3090驻留。IQ3两片GGUF约70.63GiB，其中PLE约26.82GiB，专家arena约39.97GiB；实际混用IQ2/3/4与Q2/Q4等格式，不可当均匀3位或直接换成W4 nibble。元数据见local-gguf-metadata.json。

a2当前单DUO、两颗310P3、可见44277+43693MB约85.91GiB；主机31GiB RAM，空闲可用约11GiB、swap近满。两个35B实例都在运行，不能将剩余NPU内存当成整张卡总容量，也不能叠加启动Qwen3.8。本文没有修改现有服务。

## 方案优先级

1. 优先复用社区OpenSensor Qwen3.8 310P模型实现和W4 checkpoint，验证TP2/EP2及已有lazy_shard PLE路径。按模型卡总payload181637185528字节减约102.40GB浮点PLE，非PLE约73.8GiB，双芯片容量算术余量约12GiB；不是实测峰值。需逐rank检查专家/共享层/MTP复制、运行时打包和加载峰值，再决定短上下文是否能装入。
2. 若PLE冷读成为主要瓶颈，可引入本机GGUF的IQ4_NL PLE按行解码和有限host cache；不同checkpoint的索引/张量/词表/GSQ变换一致性必须验证。这属于另一组量化质量对照，不应宣称与原浮点PLE精度相同。
3. 若原生W4非PLE仍超预算，再研究IQ专家存储+tile解码或缓存到NPU计算。逐层解码成FP16中间张量不能称原生IQ3矩阵计算；常驻W8展开会增大内存。完整Strata Ascend移植排在后面，涉及GPU内存/流/事件/Graph、dense/GDN/QSA、混合IQ专家、双NPU调度、MTP和Prefill。

本地Strata CMake列出CUDA/HIP/SYCL等后端，未发现可直接启用的Ascend后端；所附llama.cpp ggml-cann的supports_op对310P的量化Q4_0/Q8_0矩阵路径返回false，也未列IQ3_XXS支持。存在CANN目录不代表能承接Strata自定义CUDA引擎。

## 已确认的社区能力与限制

[模型卡](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i)实测为两张DUO、四颗310P3的TP4/EP4，不能照抄成TP2。checkpoint约169.16GiB含PLE，不能全部驻留a2。模型架构qwen4_exp，48层、hidden2560、512专家/top10、expert640，含QSA、GDN、PLE与gated residual；不是Qwen3.6 35B同架构换权重。

[社区ngram_embedding.py固定版本](https://github.com/opensensor/vllm-ascend/blob/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032/vllm_ascend/models/qwen4_exp/ngram_embedding.py)已包含AscendPLELazyShardEmbeddingMethod：按需mmap Safetensors行、无整表host分配，并对lazy transport跳过整表host预算检查。这意味着32GiB host不必先装下95.43GiB浮点PLE，但冷磁盘、页缓存抖动和多rank重复读取需实测；不能仅凭mmap判定性能可用。

[Strata机制说明](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md)可作分层思路参考，不将其3090 tok/s当成310P预期速度。以上网址由HTTP获取核实，搜索工具当日不可用。

## 最小实际验证顺序

先读取完整Safetensors索引做逐tensor/逐rank容量审计，明确短上下文、文本、c1、MTP0的最小预算；只下载最少测试分片。验证PLE lazy读取、真实专家矩阵、QSA及TP2切分，记录加载峰值。预算通过再按a2隔离实验环境下载所需权重，停当前服务后启动TP2短窗口生成。然后MTP1/Graph、8K/32K/112K速度与质量，最后考虑MTP2、KV量化、视觉或更长窗口。任何一步内存/质量失败都保留原始原因，不先承诺长窗口和吞吐。

推荐结论：值得研究单DUO运行该模型，优先社区310P实现+TP2容量适配，暂不完整移植Strata。已知事实证明有路径基础，尚无真实TP2服务验收。
