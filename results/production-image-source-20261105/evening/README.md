# 生产镜像启动复现与冻结门禁复核（2026-11-05）

六臂 preexec 均约 29 GiB available，模型加载时日志为 14.44–14.58 GiB，均禁用 auto-prefetch；未复现旧 B 的 20.49 GiB 内存读数；旧日志也有禁用 auto-prefetch 提示，未证实分支不同。

本轮未重建镜像，按 B、Control 交替各三次仅启动，再根据结果执行同轮生产 A、A′ 的门禁。候选及生产不可变引用见 adjudication.json。旧冻结文件未修改。

## 六臂启动结果

| 臂 | 健康 | 启动秒 | Available RAM GiB | worker PID | ExitCode | 崩溃栈 SHA256 | 内核 segfault at |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| B1 | True | 211.72 | 14.44 | 567/574 | 0 | 无崩溃 | 0 条 |
| Control1 | True | 224.53 | 14.54 | 568/576 | 0 | 无崩溃 | 0 条 |
| B2 | True | 211.75 | 14.50 | 567/575 | 0 | 无崩溃 | 0 条 |
| Control2 | True | 213.95 | 14.58 | 567/574 | 0 | 无崩溃 | 0 条 |
| B3 | True | 214.21 | 14.45 | 567/574 | 0 | 无崩溃 | 0 条 |
| Control3 | True | 212.98 | 14.48 | 567/574 | 0 | 无崩溃 | 0 条 |

B 与 Control 均为 0/3 失败。未复现稳定的候选特有启动缺陷，候选三次均可稳定启动，依派发分支进入门禁。六次均无崩溃，故不能从本轮给出内核 fault 地址/error 码，也不能确认旧故障的主机根因。ExitCode 为健康后主动停止的结果。PID 与 RAM 仅作为现场记录，不据此归因。

每臂完整 `podman logs` 单独保存；同时保存 preexec、launch-command、manifest、inspect、起止时间、退出码、内核启动区间日志及日志 SHA256。原始记录在仓库外，公开摘要仅含证据摘要及散列。旧 B 的历史日志无法追溯补采，本轮三次 B 的完整日志已分别补齐。

旧 emergency 日志复核 SHA256 a713508e…：第98行先段错误，第159行另一 TP0 才开始加载，第166行 Available RAM 20.49 GiB，第167行明确打印 Auto-prefetch is disabled。因此“旧 B 未打印该行”的线索不成立；该分支不是已证实的镜像差异。TP1 未打印开始加载，不应把 TP0 后续行为误作崩溃子进程的阶段。

## 同轮噪声与旧冻结阈值

| 指标 | 同轮 A′/A | 旧冻结阈值 | passed |
| --- | ---: | ---: | --- |
| top1_disagreement_rate | 0.0188679245283 | 0.0280373831776 | True |
| mean_conditional_topk_kl | 0.0285316708835 | 0.0372147681527 | True |
| max_conditional_topk_kl | 6.75243342666 | 6.19170636567 | False |
| max_absolute_target_logprob_difference | ≥13.3013935387 | 12.7878983021 | False |
| mean_top1_divergent_suffix_fraction | 0.814285714286 | 0.872321428571 | True |
| max_top1_divergent_suffix_fraction | 1 | 1 | True |
| mean_greedy_divergent_suffix_fraction | 0.814285714286 | 0.872321428571 | True |
| max_greedy_divergent_suffix_fraction | 1 | 1 | True |

窗口结果：`frozen_thresholds_not_applicable`。5 处目标 token 缺失于 top-20，最大绝对 logprob 差只是下界（≥13.3013935387）；下界已超过阈值，可以明确判 False。不可判定的右删失应保持 null，不视为通过。噪声超阈值即停止，不重冻或放宽阈值。

冻结阈值不适用于本窗口噪声。依规则停止，未启动门禁 B；启动复现 B 不计作门禁数据。

## G2、负载画像与限制

| 项目 | A | A′ | B |
| --- | ---: | ---: | ---: |
| GSM64 | 63 | 63 | 未执行 |
| fields_C1 | 13 | 13 | 未执行 |
| fields_C3 | 13 | 13 | 未执行 |
| short_before | 8 | 8 | 未执行 |
| short_after | 8 | 8 | 未执行 |

八格基准未执行，无会议回放、RAG 8K–32K Prefill 或通用 1K–32K × C1/C4 的收益、不劣化结论。未执行 G0、未分析或改动 Ascend 算子实现。旧 CPU 文件一致性结论保持：16869 个源码/安装文件一致，非缓存差异 0；本轮未重复重解析镜像。

窗口起点按新规则停止 active 原服务；结束未重启任何服务，没有旧端口 health 握手。窗口关闭自检通过：无运行容器、两个服务均 inactive，持锁 4188.02 秒，未超过三小时。

命令采用既有生产副本启动方式；门禁工具为 `compare_token_logprobs.py`、`eval_gsm8k_http.py`、`check_action_state_regression.py`、`check_cache_request_reuse.py`。临时控制脚本语法校验通过，右删失判定四个用例通过。原冻结文件 SHA256 `870c0b398c23c9804605f6170f01f5de86521f8c979bbc94b8a62848246dc0e0` 保持不变。
