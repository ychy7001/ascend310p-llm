# MindIEQ / 原始 W4A8+C8 交替对照（2026-10-14）

这是目前最接近产品决策要求的一轮对照。a2 两颗同型号 310P 分别运行 MindIEQ E5 服务（`18024`）和关闭 MindIEQ 的原始 W4 服务（`18028`）；每个 prompt 长度先分别 warmup，正式请求在两个服务之间交替执行五次。两边均为 TP1、W4A8、动态 C8、GDN、MTP2、Decode Graph、prefix cache off、输出 128 token，使用同一批 tokenizer token ids。Prometheus 阶段计时，采集器在收到 usage 后结束流。

| prompt | MindIEQ Prefill 中位 | 原始 W4 Prefill 中位 | 增量 | MindIEQ Decode 中位 | 原始 W4 Decode 中位 | 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 620.27 tok/s | 503.09 tok/s | **+23.29%** | 37.13 tok/s | 37.15 tok/s | -0.06% |
| 32256 | 541.99 tok/s | 449.07 tok/s | **+20.69%** | 29.17 tok/s | 29.31 tok/s | -0.48% |

所有 20 个正式请求 usage 均为完整的 prompt + 128 completion，无 OOM 或请求错误。原始逐请求数据和最终 summary 在 `paired-ab.out`；两服务日志分别为 `mindieq-service.log` 和 `original-service.log`。

## 结论

在当前固定的 TP1/310P/C8/GDN/MTP2 路径中，MindIEQ 对 8K～32K 长 Prefill 的收益已经达到稳定的 20% 以上，并且 Decode 退化小于 0.5%。这满足“值得继续投入”的性能门槛，也支持把 MindIEQ 作为长 Prefill 的产品候选开关。

仍有两个边界：两条服务分别占用两颗同型号芯片，不能排除芯片间温度或微小频率差异；本轮没有测 64K/112K、并发、P95 或完整质量集。因此建议先将 MindIEQ 默认阈值保持为长 Prefill 分流（当前 128 token），继续做 64K/112K 与质量门禁，不把 Decode 端宣称为加速。
