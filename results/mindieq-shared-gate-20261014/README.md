# MindIEQ Gate/Up token级量化共享（2026-10-14）

原路径先复制top8 FP16输入再量化；新路径每token量化一次，AscendC gather按设备端sorted_tokens复制INT8码与FP32 scale，并生成16零padding行。Down输入随expert不同，保持原准备方式。主模型权重和MatMul算术不变，Decode保持native。

## 真实NPU微测试

芯片1，真实layer0专家0..7 Gate/Up矩阵N1024/K2048，合成FP16激活、固定分组top8，tokens17/32/128/512。warmup5/重复30交替，两种完整准备及Gate/Up投影都计入host dispatch+synchronize，不是纯kernel latency。

| token/route rows | 原准备ms | 共享准备ms | 原Gate/Up ms | 共享Gate/Up ms |
|---|---:|---:|---:|---:|
|17/136|0.140|0.129|0.476|0.464|
|32/256|0.183|0.141|0.516|0.476|
|128/1024|0.303|0.184|0.941|0.846|
|512/4096|0.999|0.389|3.406|2.794|

所有A8码/scale及投影输出逐位一致，4案例×8连续异步投影通过；Graph重放同时更新输入和路由索引通过。512token Gate/Up耗时减少约18%，不能外推整模型同幅收益。真实专家数仅8，未覆盖256专家路由调度或全部层的模型质量。

初版torch INT8 index_select在Graph capture走aclop GatherV2失败，失败日志保留。改独立AscendC gather，INT8行复制与8行一组scale写回满足32B对齐，多分配ceil8+16行，返回routes+16有效view。q/s/indices/output/scale都recordStream，RunOpApiV2不捕获Tensor，复用原算子的异步生命周期方式。

## 独立可选服务门禁

新库SHA256 9fd97f447613997189e2a473d3e374cf485275d60e376b50cf7c1c8ca50595ba，grouped-shared-build-v1目录；旧库不覆盖。QWEN36_MINDIEQ_SHARED_GATE=1需同时开启custom_quant及MindIEQ；默认off。旧库与默认代码路径仍兼容，缺少gather的新开关不能用于旧库。

E5服务保持D4 TP1/48G/W4A8/C8/MTP2/Decode Graph/GDN/4096:512/112K/.95/maxseq1/prefix off及阈值128；仅更换新库、开启共享准备。服务已开始加载，启动日志custom_quant=True/min_tokens128/shared_gate=True；未宣称整模型门禁通过。

build.sh使用全新SOURCE_DIR/BUILD_DIR；run-micro.sh、start-service.sh、run-service-gates.sh记录实验机命令，复现需替换路径。控制器先health/sanity、四业务输入，再8K/32K各三次计时与MTP、边界题、固定64题。微测试输出shared-gate-v1-20261014.jsonl及构建、失败日志均保留，实际配置与制品见preflight.json/artifacts.json。

## E5已完成性能阶段

实际服务启动、C8规划、Decode Graph及sanity5/5通过。相同prompt SHA、128输出、warmup1/repeat3：8K Prefill624.768 vsD4 607.457tok/s（+2.85%），Decode34.680 vs34.821（-0.40%）；32K Prefill546.618 vsC3 533.057（+2.54%），Decode28.961 vs28.953（+0.03%）。两种长度输出各3/3全文一致，8K接受率仍92.963%、32K100%。32K参考C3阈值17而E5为128，真实Prefill/MTP本测试形状远高于阈值，仍明确保留配置差异；不是本轮新D4 32K对照。

四类业务和边界题已完成，64题仍运行，不以完整门禁脚本存在代替测试完成。phase-comparison.json仅分析性能阶段，analyze-service.py要求DONE和全部结果才生成完整统计。本次模型Prefill+2.5%～2.9%不同于单层Gate/Up约18%收益，不能混用两种口径。

## E5完整服务门禁结果

控制器已terminal，state DONE，固定64题E5/D4均63/64，数字答案变化0、全文37/64一致，共同205输出512上限截断。基础sanity5/5，711边界正确。完整服务原始输出在service/，analyze-service.py核对请求完整、输入SHA和质量协议后生成service-comparison.json。

会议输入三次全文与D4一致（145输出/stop），TTFT1.921→1.871秒（约-2.6%），Decode39.180→38.918tok/s（约-0.7%）；短对话/代码/分析保持原后端分流，TTFT约0.392/0.336/0.385秒，部分文本有变化，这类场景大多达到256输出上限，不作为完整业务质量评分。固定小子集无数字答案回归不证明全面精度等价。

工程结论：共享Gate/Up量化有约2.5%～2.9%端到端Prefill收益，可与阈值128一起保留为可选实验候选。8K接受率92.963%及相对原native Decode下降未解决，默认总开关和shared_gate均off。E5尚未重测近112K检索；C3/D4长上下文证据不能替代E5组合验证。TP2 C8/32G/多并发/多模态/全面质量仍未覆盖。

本轮只读a1参考完成，数据见../a1-w8-reference-20261014。通过cache_salt并要求prefix命中0，避免32K复用此前8K前缀；跨机器/TP/软件/KV/MTP比较不能当纯量化加速比。
