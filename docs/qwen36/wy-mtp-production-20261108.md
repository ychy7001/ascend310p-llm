# TP2 W8A8 + WY + MTP2 满上下文生产配置

本页记录2026-11-08的配置收尾，当前设备验收进行中；以最终归档状态为准，不提前宣称生产放行。a1只读参考，实验在a2运行。

## 目标与基线

a1运行manifest已归档到`results/wy-mtp-production-20261108/a1-reference-manifest.json`：TP2、max_model_len262144、seq4、batch4096、memory0.9、prefix caching开启、异步调度。模型服务名为`Qwen3.6-35B-A3B-w8a8`，保留reasoning parser、tool parser和视觉路径。新配置以冷Prefill为目标，默认关闭prefix caching，使用独立cache salt并验证无命中。

a1/a2均为双310P3，显存容量相同；驱动分别26.0.rc1和26.1.1。保持已验证W8镜像的CANN9.1.0-beta.1、torch_npu2.10.0.post2及WY kernel，不统一W4底层栈。

## 新暴露的问题及修复

1. `w8-production-full-v1`：262144/seq4/batch4096启动，第四个主模型Graph捕获时EL0008 event resources不足。收敛MTP2布局到`[3,6,12]`，三请求批次使用标准padding，随后二/三/四并发确定JSON通过。
2. `w8-production-full-v2`：4K测试通过，8K时兼容适配器在普通Prefill也生成4096×262144固定列宽mask，额外2GiB分配失败。修复要求forward context实际为FULL且与wrapper一致，普通eager/Prefill直接走原有路径。数学mask定义及WY kernel不变；适配器SHA更新并单列，旧候选的114688/seq1验收范围不能外推到262K。

3. `w8-production-full-v3/v4`：主模型Graph下双请求MTP触发同一Mul设备异常，关闭async仍复现；未证明精确根因，不作为生产默认。
4. `w8-production-full-v5`：主模型eager、草稿eager，保留native WY与MTP2。二/三/四并发JSON、四并发128输出12/12及混合输入输出4/4通过；128K+128完成（TTFT175.075秒，Prefill748.933 tok/s，无prefix命中）。262K边界与后续PDF仍在执行。

## 验收定义

- 4K/8K/16K/32K冷文本：每档warmup一次，正式三次，以请求阶段计数器计时，prefix hits必须为0；不称为纯kernel吞吐。
- MTP：确定JSON，多轮连续生成，二/三/四并发布局；单请求4K/128输出，四并发三轮，以及输入/输出长度不等的混合并发。
- 容量：131072+128、262016+128精确token预算请求；输入输出合计262144，额外超限请求应返回400。该测试证明容量和执行完整性，不证明262K检索语义质量。
- PDF：四页功能检查，单列截断与finish_reason；不宣称真实OCR业务非劣效。
- 制品：最终镜像默认入口无host源码挂载，独立启动及请求验收；按registry digest核验manifest/config，并独立清场。

`max_num_seqs=4`是调度并发上限，不承诺四条262K请求同时驻留。首轮KV pool报告635026 tokens（约2.42条262K请求），以最终配置启动日志为准；超出驻留容量由调度器处理，未测试的极端组合不写成已验证。

## 部署资产

- `deploy/qwen36-wy-mtp-production.manifest.json`：保留a1模型路径、端口18081及服务名，调用镜像内入口。
- `deploy/qwen36-wy-mtp-production.container`：a1路径的Quadlet模板，显式覆盖ENTRYPOINT以执行现有manifest launcher；两卡设备、60秒冷却、失败重启均写入配置。
- `deploy/qwen36-wy-mtp-image.env.example`：通用默认入口部署环境，交付时更新固定镜像引用及验收过的容量参数。

a1现有Quadlet由llmctl生成。实际部署时需同步部署描述中的镜像与启动配置，避免重新渲染恢复旧值；本轮不修改a1。切换前保存原manifest/Quadlet和旧镜像digest，停服务、确认NPU空闲、冷却至少60秒后再启动；回滚同样执行冷却。不能只换manifest而继续使用旧镜像，也不能将旧Quadlet的Exec直接附加到新镜像ENTRYPOINT后。
