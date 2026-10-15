# RC/EP Decode 瓶颈分析

## 对照结论

同一模型、TP1、MTP1、Graph、GDN、MindIEQ、独占请求和相同 prompt：RC 与 EP 的 Prefill 差异约 2%，但 Decode 差异约 14%--22%。大块 memcpy 和大矩阵微基准接近，因此不能把差距简单归因于理论带宽或 AI Core 峰值。

当前 RC 服务还确认了明显的 host/runtime 参与，但 CPU 利用率不能直接等价为同步等待。下一步应以 stream/task 时间和算子间空洞为依据，而不是继续调整 CPU affinity。

## RC 设备 profiler

来自 `results/orangepi-deployment-20261014/profile-decode-full/op_statistic_20261014165717.csv` 的一次 Decode 窗口（C8、MTP1、Graph、GDN、MindIEQ）：

| 算子类型 | 调用次数 | 总时间 | 占比 |
| --- | ---: | ---: | ---: |
| MatMulV2（含普通 FP16/NZ） | 967 | 108.75 ms | 45.66% |
| QwenW4A8Int4MatmulV310 | 276 | 29.19 ms | 12.25% |
| NotEqual | 12 | 18.26 ms | 7.67% |
| TransData | 1665 | 16.93 ms | 7.11% |
| paged_attention_decoder_mask_0 | 39 | 16.51 ms | 6.93% |
| C8 cache_to_nz | 39 | 9.51 ms | 3.99% |
| RecurrentGatedDeltaRuleV310 | 103 | 7.42 ms | 3.11% |

单次平均时间约为：

- `paged_attention_decoder_mask_0`: 423 us
- `c8_cache_to_nz_kernel`: 244 us
- W4 自定义算子：约 106 us（不同 shape 的平均值）
- `NotEqual` 中有 3 次约 4.4--4.9 ms 的长任务，值得结合 stream 9 的前后 task 继续定位

这解释了为什么 C8 单算子优化不能直接转化为同等比例的 Decode 提升：C8 转换约占该窗口 4%，而 MatMul、掩码、TransData 和调度链占更大部分。

仓库中已经对 C8 转换做过三种真实 310P 长上下文候选门禁：流水线双缓冲在 114K 将转换从 3.258 ms 增加到 3.481 ms；连续页分配为 3.350 ms；K/V 分核为 3.278 ms。三者均通过数值/Graph 检查，但没有稳定加速证据，因此没有进入主库或服务制品。直接 C8 Attention 的独立实测也显示 114K 为 27.886 ms，而原生 PA 为 3.216 ms。当前瓶颈需要改变数据流或使用新的硬件原语，继续堆叠 tile/core 调度开关没有工程收益。

## 优先级

1. **先定位掩码和 NotEqual 的等待关系。** `paged_attention_decoder_mask_0` 没有普通 shape 信息，且固定约 423 us；需要把 task timeline 与 stream/event 对齐，确认它是实际计算、同步后的尾部任务，还是 Graph replay 中的固定开销。
2. **减少 Decode 的 TransData/小张量链。** 当前 TransData 总计约 16.9 ms，device block index 候选只改善约 0.37%，说明单独消除 `seq_lens.cpu()` 不是主因；应继续检查 block table、mask、Q/K layout 是否能在 Graph 内复用。
3. **继续优化 W4 小 GEMM shape。** W4 自定义算子已占 12.25%，它是 RC/EP 都会付出的主要工作；应按真实 shape 分组测试，而不是只看大矩阵带宽。
4. **保留 persistent NZ 为实验选项。** range update 已把 C8 转换从约 3384 us 降到约 162 us，但端到端 Decode 尚未证明稳定提升，不能用它掩盖前面三类瓶颈。

## 长上下文 profile 证据

仓库中的真实 Graph profile 对比来自同一套 E5 制品和 MTP2 路径；profile 在首个流式文本之后开启，可能漏掉最初 Decode 步，因此这里只比较捕获窗口的 kernel 求和，不当作墙钟占比。

| 输入/输出 | Paged attention mask | C8 conversion | MatmulWeightNz | W4 expert |
| --- | ---: | ---: | ---: | ---: |
| 8192/128 | 6.55% (0.415 ms/call) | 3.87% (0.245 ms/call) | 47.12% | 15.70% |
| 114560/128 | 36.77% (5.544 ms/call) | 21.78% (3.284 ms/call) | 20.23% | 6.79% |

长上下文 profile 中，mask 与 C8 conversion 合计 `58.55%` 的捕获 kernel 时间；普通 MatMul/W4 的单次平均时间基本不变，份额下降主要是历史 KV 处理变重。该结果支持以下工程判断：

进一步按原始 kernel 明细拆分后，`paged_attention_decoder_mask_0` 的 480 次调用分布在 stream 6/7/72，单次耗时中位数 5.544 ms、P99 5.555 ms，等待时间只有约 37 us；它不是 host 等待造成的假占比，而是稳定的设备端工作。C8 转换的 492 次调用分布在相同 stream，单次中位数 3.284 ms。相反，163 次 `NotEqual` 的中位数只有 2.813 us，主要长尾为 stream 30 上的 3 次约 3.2 ms 调用；总量约 10.8 ms，远小于长上下文 mask 和 C8 转换，不能作为首要优化目标。

- `QWEN36_C8_DEVICE_LENGTH=1` 对短请求有价值，但接近最大上下文时有效页数接近全量，不能解决长请求的主要成本。
- persistent NZ/range update 只减少“新增长度所在页”的转换，不能消除首次长历史的 full conversion。
- 长上下文的高价值方向是复用原生 paged attention 的 mask/reduction 路径，同时减少 C8 转换的重复工作；当前 Direct C8 Attention 原型已经在 4K/32K/114K 上证明比原生 PA 慢，不能接入默认路径。任何新的 mask 或转换方案都必须通过 32K/112K 真实生成和数值门禁后才考虑接入默认路径。

原始摘要与恢复验证见 `results/decode-profile-20261014/`、`results/decode-profile-112k-20261014/`。两个服务在 profile 结束后均恢复并通过 sanity，profile 本身不用于宣称产品吞吐。

对当前稳定配置（开启 `QWEN36_C8_DEVICE_LENGTH=1`）的 114560/128 完整 profile
请求也已成功完成，Graph replay 和 MTP 接受率均正常；但 CANN 在停止 profiler
时导出文本失败（`ERR04005 PROF internal error`），没有生成可解析的 kernel CSV。
这次请求只作为配置/请求恢复证据，不替代旧 profile 的 kernel 占比，也不把 profiler
改变后的墙钟时间当作性能数据。原始请求和错误日志在
`results/profile-current-114k-20261015/`。

## 当前产品决策

稳定默认配置保持 C8 device-length + MTP1 + Decode Graph。`RC_DEVICE_BLOCK_INDICES=1`、`QWEN36_C8_REUSE_INT32_METADATA=1` 和 persistent NZ 继续作为诊断/实验开关；已有真实数据没有显示足够稳定的收益来默认打开。下一轮最有价值的实验是针对 mask/NotEqual 的 stream timeline，而不是重新做相同的 affinity 或 memcpy 微基准。
