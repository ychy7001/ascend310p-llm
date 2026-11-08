# 三臂门禁尝试：生产比较异常，候选未启动（2026-11-05）

窗口 `day-20261105`，计划顺序 A → A′ → 冻结 → B。七项按 case 归一化判定，倍率1.25，无绝对地板。本轮未修改判定工具、控制器、holder或模型启动参数，八格基准未执行。

最终状态：`failed`。汇总脚本正常返回失败摘要（exit0）；B未采集时，不执行依赖B的比较重算/冻结先于B校验，逐臂完整日志与原始数据保留在仓库外，散列见 adjudication.json。

失败发生在 A/A′ 比较：`ValueError('No shared top-k support')`。`replay-21` 第2个生成位置，两臂条件前缀相同，返回的20个token集合交集为0。比较工具异常退出，具体阈值未冻结、B未启动。未修改工具、添加替代值或重新开窗口。

注意：实际控制器在当轮A/A′后直接冻结并判B，源码没有派发描述的双向噪声预检分支；本轮未自行增加。

## G1 判定

| 指标 | A′/A 噪声底 | B/A | 阈值1.25× | passed |
| --- | ---: | ---: | ---: | --- |
| top1_disagreeing_case_rate | 比较异常，未产出 | 未启动 | 未冻结 | 未判定 |
| mean_case_top1_mismatch_rate | 比较异常，未产出 | 未启动 | 未冻结 | 未判定 |
| mean_case_mean_conditional_topk_kl | 比较异常，未产出 | 未启动 | 未冻结 | 未判定 |
| mean_case_max_conditional_topk_kl | 比较异常，未产出 | 未启动 | 未冻结 | 未判定 |
| mean_prefix_absolute_target_logprob_difference | 比较异常，未产出 | 未启动 | 未冻结 | 未判定 |
| mean_top1_divergent_suffix_fraction | 比较异常，未产出 | 未启动 | 未冻结 | 未判定 |
| mean_greedy_divergent_suffix_fraction | 比较异常，未产出 | 未启动 | 未冻结 | 未判定 |

## 诊断：只报不判

下表不增加判定条件；与七项判定表重合的均值项仍以判定表为准。最大目标 logprob 差有删失时仅为下界。

| 旧诊断指标 | A′/A | B/A |
| --- | ---: | ---: |
| top1_disagreement_rate | 未产出 | 未产出 |
| mean_conditional_topk_kl | 未产出 | 未产出 |
| max_conditional_topk_kl | 未产出 | 未产出 |
| max_absolute_target_logprob_difference | 未产出 | 未产出 |
| mean_top1_divergent_suffix_fraction | 未产出 | 未产出 |
| max_top1_divergent_suffix_fraction | 未产出 | 未产出 |
| mean_greedy_divergent_suffix_fraction | 未产出 | 未产出 |
| max_greedy_divergent_suffix_fraction | 未产出 | 未产出 |

| 只报不判的计数 | A′/A | B/A |
| --- | ---: | ---: |
| target_logprob_censoring | 未产出 | 未产出 |
| positions_above_kl_0.5 | 未产出 | 未产出 |
| positions_above_kl_1 | 未产出 | 未产出 |
| positions_above_absolute_error_0.5 | 未产出 | 未产出 |
| positions_above_absolute_error_1 | 未产出 | 未产出 |

## G2

| 项目 | A | A′ | B | B−A绝对差 | B−A′绝对差 | passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GSM64 | 63 | 63 | 未执行 | 未执行 | 未执行 | 未执行 |
| fields_C1 | 13 | 13 | 未执行 | 未执行 | 未执行 | 未执行 |
| fields_C3 | 13 | 13 | 未执行 | 未执行 | 未执行 | 未执行 |
| short_before | 8 | 8 | 未执行 | 未执行 | 未执行 | 未执行 |
| short_after | 8 | 8 | 未执行 | 未执行 | 未执行 | 未执行 |

## 逐臂启动与收尾

| 臂 | 健康 | RAM GiB | worker PID | ExitCode | 崩溃日志散列 | 内核segfault at |
| --- | --- | --- | --- | ---: | --- | --- |
| A | True | 14.54 | 567/574 | 0 | 无崩溃 | 0 条 |
| A_prime | True | 14.52 | 567/574 | 0 | 无崩溃 | 0 条 |
| B | 未启动 | 未采集 | 未采集 | 无容器 | 未发生 | 未运行 |

## 时延账

排队 0.361秒，持锁窗口 2839.206秒（47.32分钟）。

| 臂 | 容器总秒 | 启动至G1秒 | G1秒 | 短题前测秒 | GSM64秒 | 结构化秒 | 短题后测秒 | G2合计秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1413.290 | 213.463 | 509.937 | 8.493 | 423.896 | 246.506 | 6.507 | 685.402 |
| A_prime | 1407.812 | 212.313 | 507.655 | 8.228 | 421.680 | 246.969 | 6.438 | 683.315 |

B未启动，无臂时长或G2时间，不能填作测得的0秒。容器时长来自 inspect 的 StartedAt/FinishedAt；G2工具时长为本地 status 开始到最终输出文件 mtime，未计进程退出的微小尾部开销，不能当作精确子进程壁钟。状态观察器未改变控制器。

下一轮八格三臂窗口的串行token工作量估算 98.98分钟（含15分钟清理余量），落在三小时。基于每个长度最慢的同长或略长G1 case、三次启动、C4按四个串行C1计费，24次预热也按完整256-token请求计费；未假定C4提速。C4争用及各长度streaming耗时尚未实测，这只是排期估算。建议下一轮预留120分钟窗口（含至少15分钟清理时间），给99分钟估算额外余量；执行前仍须先完成门禁。

## 负载收益与限制

本轮尝试精度/功能门禁，无会议回放、RAG 8K–32K Prefill 或通用1K–32K×C1/C4的TTFT/TPOT/吞吐收益结论。基准未执行，完整设备/性能验收尚未完成。top-k KL仅限共同显式支持；生成观测只比较相同条件前缀。只在分歧前同选前缀采量级判定，诊断极值不参与门禁。

本轮不能判定候选，G2只取得A/A′计数，没有B，不能声称三臂通过。后续需要先处理本轮比较异常并完成门禁，再安排基准。本轮估算只供排期。

窗口结束前自检：运行容器为空，两个服务inactive，未重启服务或执行旧端口health握手，窗口未超过三小时。未分析或改动算子/内核，未操作其他仓库、未推送Git或merge。
