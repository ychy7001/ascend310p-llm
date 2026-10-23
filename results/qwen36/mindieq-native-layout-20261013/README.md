# MindIEQ native权重布局原型（2026-10-13）

在既有设备端grouped原型上，将GM读取调整为native signed W4排列；在UB完成重排与XOR0x88等价变换，复用MindIEQ的biased解包、INT8 Cube和校正路径。没有完整专家权重全局反量化，不要求第二套W4专家权重常驻。外部实现固定d9c9871aa335f7b5638e9a2679a45bb3e7561d3a，许可与来源沿用前一轮；生成外部头和二进制不进入本仓库。

## 真实设备门禁

310P3/CANN9.1，12个单投影案例全部通过，包括零专家、单行与不规则尾块、三份实际路由counts及动态counts Graph。每个案例8次连续提交输出通过；CPU整数参考rtol/atol=.002/.002。详见mindieq-native-layout-v1.jsonl及txt。

真实layer0全部256专家权重的完整MLP，三份真实counts，warmup5、20次交错测量；包含激活量化、SwiGLU、padding及两投影，不含router/shared expert/attention/token combine。本轮native布局测量一次，不能声称两轮可复现。p05/p50/p95为路由counts选样，非耗时分位。

| counts | 现有W4 ms | native布局MindIEQ grouped ms | 耗时减少 |
|---|---:|---:|---:|
| p05 |19.176|13.262|30.8%|
| p50 |19.044|13.159|30.9%|
| p95 |17.209|10.366|39.8%|

四后端×8连续提交×3案例=96输出通过；误差与不同激活量化器有关，不代表模型质量验收。native布局额外片上变换比先前biased布局慢，换取服务无需双份权重。W8为逐专家Python调用对照，不代表最佳W8 grouped实现。

## 可选服务接入

experimental/mindieq_prefill_adapter.py只替换TP1大于16 token的grouped routed expert路径；现有native Decode、共享专家、BF16 MTP及C8/Graph路径保留。仅支持本仓库SSZ对称per-channel权重（G128元数据重复scale），不支持任意独立group-wise/非零offset checkpoint。当前为实验选项，默认关闭。

开启时在run-candidate.sh之前设置QWEN36_MINDIEQ_ENABLED=1及MINDIEQ_LIBRARY为编译产物的宿主机绝对路径。先运行--check核对hash。此开关不改变生产默认，也不声称TP2或Prefill Graph支持。

复现依次执行build-native-layout.sh、run-native-layout.sh、run-native-layout-real-mlp.sh；停止芯片0上的模型容器后再运行microbench，不能同时初始化该芯片。路径需替换为自己的work/release目录。start-native-layout-service.sh为本次服务启动记录。端到端结果见下文。

固定seed310、同dataset SHA/32题/512输出上限的GSM检查：与既有GDN/4096候选均31/32，同一道题截断未提取答案，最终答案变化0，全文一致7/32。数值路径不同导致措辞变化，不能声称全精度等价。基本生成sanity5/5通过。原始输出与quality-comparison.json均保留。初测bench_http_stream提示词与旧基线不同，不能用于严格速度比较；随后使用同一bench_server_phases协议复测。

同一bench_server_phases协议8K/128三次完成：Prefill中位629.615tok/s vs旧候选503.226（+25.1%）；Decode35.655 vs37.079（-3.8%）。三条输入SHA完全匹配，输出全文1/3一致。测量基于服务phase counter，warmup1/repeats3、prefix off、单流；参考是同日旧结果，尚非交替A/B。初测HTTP提示词不同得到686tok/s，不参与此结论。

32K（32256输入/128输出）三次同协议完成：Prefill中位550.368 vs450.652tok/s（+22.1%），Decode29.070 vs29.174（-0.35%）。三条输入SHA完全一致，输出3/3全文一致。总结原始基线中位数时以recheck-p8192-o128.jsonl为准（8K 503.226/37.079），其他恢复批次507.211/36.944不混用。分析复现python3 results/qwen36/mindieq-native-layout-20261013/summarize.py。


114560输入token的近112K检索通过，输入SHA与旧候选一致，返回ASCEND732941，14输出tokens。耗时296.351s vs旧候选348.941s，降低15.1%；这是单次检索canary的总请求时间，不能当作112K持续Decode速度或全面长文质量。长请求后sanity5/5通过。实际服务仍在18024运行，实验开关显式开启；仓库默认关闭。

工程结论：MindIEQ设备端grouped+native布局已出现模型Prefill收益，值得进入配对A/B与更广质量测试。8K Decode轻微退化需要后续排查；不宣布稳定产品默认或超过W8A8。当前112K能力上限不变，本轮未测试更大上下文、并发P95、多模态、TP2 C8或32G。计时均保留原始结果，外部原始代码/二进制未提交。
