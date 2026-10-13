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
