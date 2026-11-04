# 字段与长上下文配对验证（2026-11-04）

最终判定见 `adjudication.json`：结构化字段门禁通过，原自由文本业务语义仍未接受。

- A为原生产9.1-beta/post2，B为固定9.2.1/post7/0.30＋PR17640配套H/O；均原Torch WY、35B-A3B W8A8 TP2。整体栈对照不能做单组件归因。
- 每栈26条结构化请求全部通过，52条合计；长输入覆盖4095、4096、4097、16384、32768 chat tokens，C1/C3各一轮。输入token ID哈希和usage计数全部匹配。
- 每栈9条原始自由文本请求完全配对；`A-manual-review.json`、`B-manual-review.json`按预先冻结的`freeform-semantic-oracle.json`人工审阅。B另有当天→本周内反例，两边均有保守状态标准下的完成状态补写。
- `paired-summary.json`只汇总自动字段门禁，不能覆盖最终人工判定。`A-checkpoint.json`、`status.json`保留过程状态。CPU evaluator验证使用合成突变，不是额外NPU结果。
- 原始HTTP响应、前后短题、worker载库、镜像inspect、运行脚本及SHA均保留。未归档权重和缓存。实验耗时不能当作吞吐比较。
- `service-restoration.json`独立确认原服务HTTP200、无运行实验容器。未修改宿主CANN/驱动或a1。

不证明真实长会议/262K质量、逐token或logits等价，不构成生产采用或性能验收。
