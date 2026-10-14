# 310P GDN + MTP Issue #17856 复现记录

## 当前结论

当前 a2 服务仍使用包含 `acceptedTokenNum > seqLen` 后直接 `return` 的
`recurrent_gated_delta_rule_v310` kernel。该路径与上游
[Issue #17856](https://github.com/vllm-project/vllm-ascend/issues/17856) 一致，
尚未修复。

## 已执行的服务回归

- 服务：a2 `qwen36-tp1-48g-112k`，TP1、W4A8、C8、MTP2、Decode Graph、`max_num_seqs=1`
- 请求：OpenAI `response_format={"type":"json_object"}`，固定提示和 `max_tokens=32`
- 次数：10 次顺序请求
- 结果：10/10 返回合法 JSON `{"name": "会议", "count": 3}`，均 `finish_reason=stop`
- 服务日志：无 `ERROR`、`Traceback`；该窗口 MTP 统计为 drafted 64 / accepted 64，两个位置接受率均 1.000

这个结果只覆盖单请求、短输出、固定 grammar 约束，不能覆盖 Issue 中“上一轮接受数大于当前轮 query 长度”的变长状态场景。

## 单算子复现状态

目标输入是当前长度 `[2, 4]`、上一轮接受数 `[4, 1]`，并将每条请求的 state 槽填入可区分值，观察第一条请求的非法边界是否导致整批提前退出或后续请求状态不更新。

当前服务容器中 Python wrapper 可以看到 `num_accepted_tokens` 原样传入 v310 op，wrapper 的变长 flatten 不会过滤 `acceptedTokenNum > seqLen`。但在脱离服务的独立 Python 进程中，custom OPP 尚未完成注册，`torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310` 不可见，因此还没有把上述输入送入真实 kernel。下一步需复用服务的 OPP 初始化环境，或构建一个只加载该 OPP 的一次性诊断容器；不修改运行中的服务。

## 使用边界

普通文本接受率下降不能由本次 Issue 直接解释。严格 JSON/Schema 在专门修复并重建 310P OPP 前仍是未验收能力。修复方案应参考 [PR #17839](https://github.com/vllm-project/vllm-ascend/pull/17839) 的逐请求状态槽和边界处理思路，但需针对 310P 的 state layout 单独适配。
