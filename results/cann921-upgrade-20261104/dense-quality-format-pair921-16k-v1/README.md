# 27B排序/标点窄范围诊断

最终状态仍not_accepted_short_format_gate；原短题7/8未被诊断结果覆盖。

6类请求各3次，seed0/greedy、不同冻结cache_salt、记录raw token IDs和top5 logprobs。原排序3/3生成279，token IDs=[17,22,24,248046]，没有逗号token11；其他五类15/15符合预期：数字逗号复制、字母逗号复制、分号复制、数组复制和JSON schema排序。因此本例不是已观察到的服务解析删逗号；不能据此排除量化影响或证明基础模型本身能力上限。

首个原排序样本生成首字符2后，下个token7的logprob约-0.257，逗号约-1.491；greedy选择7。该数据仅解释这个部署的选择，不是与高精度参考的数值误差比较。JSON schema改变输出约束，不能拿其通过覆盖原题失败。

固定镜像/配套H/O/原Torch WY/16K/TP2与前轮一致，两个worker载库核对通过。未做业务/性能/GSM，未修改宿主或a1。源脚本、冻结SHA、原始响应、完整日志与原服务恢复检查均保留，以adjudication.json为判定。

恢复后的a2原35B W8A8服务完成同样18条诊断，18/18符合预期；对27B的18条逐请求比较见a3b-narrow-comparison.json。仅model字段不同，但栈和上下文配置不同，不作单变量模型能力归因。原服务最后再次健康HTTP200，见相邻a3b-original-format-control-v1/service-health-after.json。
