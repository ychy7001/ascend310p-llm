# TP2 W8A8 + WY + MTP2 满上下文部署配置

原 a2 制品 `a11b8aca…` / image ID `bb0fa66f…` 已通过完整工程验收。随后部署模板改指的 `777611…` 是另一重建 staging 制品，a1 推理 canary 失败且确认缺少原制品修复，不能继承本页历史验收结论。详见[制品一致性审计](a1-wy-mtp-artifact-audit-20261108.md)。本页不宣称正式OCR业务非劣效。

## 固定配置

| 项目 | 值 |
| --- | --- |
| 模型/量化 | Qwen3.6-35B-A3B，现有W8A8权重 |
| 并行 | TP2，两张310P3 |
| Prefill | native WY，沿用既有kernel数学路径 |
| MTP | 2个speculative tokens |
| 主模型图 | FULL_DECODE_ONLY，捕获[3,6,12] |
| Draft | eager |
| 上下文上限 | 262144，包含输入和输出 |
| 调度并发上限 | 4 |
| max_num_batched_tokens | 2048 |
| memory utilization | 0.90 |
| prefix caching / async | 开启 / 开启 |
| served model name | Qwen3.6-35B-A3B-w8a8 |

失败 staging image ID：`bb033d6b18c6f6ee342f13b1457f5c8cdfeeb90691698b3e68fa16f5535db952`；digest：`sha256:777611b2e13da9be433e2e98ce67777713d07fc3321717af16a414e2594a5189`。该制品缺少原验收修复，a1 首次推理失败，已禁止放行。当前部署候选恢复为完整验收制品 `bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4` / `sha256:a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5`。

镜像元数据版本为torch2.10.0+cpu、torch_npu2.10.0.post2、vLLM0.23.0+empty、vllm-ascend0.23.0rc2.dev93+g88eac2713，CANN9.1.0-beta.1；本地overlay另外登记SHA，不把包版本当作未修改源码证明。

## Graph与状态修复

主模型图优化已开启。draft eager来自性能筛查：主图三档/draft eager约46 tok/s；双图受event资源限制需收敛到[12]，小批次padding后连续两窗约29 tok/s，Prefill相同。该筛查不是交替A/B，不据此宣称正式百分比收益。定位和无效实验标签见[Graph定位报告](wy-mtp-graph-localization-20261108.md)。

镜像包含PR18730配套recurrent状态表修复（经源码SHA核对复用opp-review构建，eager/Graph各42项独立回归通过），并将MTP slot block IDs在乘法前提升到int64，绕过模型上下文中的int32 Mul失败调用点。WY kernel、冻结协议/规则/worker/R和CANN版本保持不变。recurrent配套OPP、wrapper和proposer的变更独立登记，不能只替换Python wrapper。

## 完整窗口实测

`main-graph-release-full`耗时1324.247秒，driver0，独立清场通过。请求阶段计数器统计，三次正式请求取中位数；该历史窗口实际关闭 prefix cache，且使用每请求独立 cache salt，prefix hits 为0；后续生产模板改为开启不属于该窗口验收范围。数字是服务Prefill阶段吞吐，不是纯kernel吞吐。

| 冷文本输入 | Prefill tok/s | TTFT秒 | 有效次数 |
| --- | ---: | ---: | ---: |
| 4K | 2049.6 | 2.002 | 3/3 |
| 8K | 1937.9 | 4.233 | 3/3 |
| 16K | 1746.8 | 9.390 | 3/3 |
| 32K | 1456.7 | 22.512 | 3/3 |

- 旧资格窗口（4K输入、128输出）：decode中位数45.877 tok/s，draft acceptance约85.1%。
- 旧资格窗口四并发端到端测试：12/12完成；含Prefill的聚合输出吞吐中位数35.703 tok/s，TTFT P50/P95为6.755/9.316秒。P95是12样本经验分位数；该数字不替代下方最新4K/512受控对照。
- 二/三/四请求确定JSON通过，混合输入/输出长度4/4完成。
- 131072+128完整返回：TTFT177.967秒，Prefill736.752 tok/s。
- 262016+128完整返回，总token262144：TTFT595.558秒，Prefill440.041 tok/s；超限请求返回400。
- 四页PDF功能检查4/4；pdf-01达到512输出上限，其余stop。长请求后JSON通过。
- 每10秒npu-smi采样，观察到的单卡最高使用41760MB；不是allocator peak。

长输入测试证明容量和执行完整性，不证明262K检索语义质量。PDF仅功能检查；多模态处理缓存未专门禁用，不称为完全冷视觉吞吐。没有通过这些测试回溯修改旧G1结果，或新增正式OCR非劣效声明。

`max_num_seqs=4`是调度上限，不承诺四条262K请求同时驻留。启动报告KV pool约635026 tokens，即约2.42条满上下文请求，超出驻留容量由调度器处理。

## Decode收益补验收

2026-11-08在a2完成同一最终镜像的MTP0/1/2受控对照，并将归档的a1现役profile在a2复现两次。每个窗口使用4K固定prompt、512输出、唯一cache salt，单并发和四并发各2轮；记录客户端TTFT/TPOT、服务端请求阶段Decode、端到端聚合吞吐和MTP接受率。a1只做了只读身份核对，没有发送请求。

| 臂 | 单流服务端Decode | 单流客户端TPOT | 四并发服务端Decode/请求 | 四并发端到端聚合 | 四并发TTFT P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最终镜像 MTP0 | 33.10 tok/s | 30.21 ms | 18.64 tok/s | 61.94 tok/s | 5.97 s |
| 最终镜像 MTP1 | 40.58 tok/s | 24.64 ms | 22.51 tok/s | 71.10 tok/s | 6.34 s |
| 最终镜像 MTP2 | 43.69 tok/s | 22.89 ms | 22.62 tok/s | 70.04 tok/s | 6.69 s |
| a1旧profile复现 | 32.28 tok/s | 30.98 ms | 17.31 tok/s | 51.60 tok/s | 10.68 s |

因此，MTP1相对同镜像MTP0的单流生成吞吐约提升22.6%，四并发端到端吞吐约提升14.8%；MTP2相对MTP0单流约提升32.0%，四并发端到端约提升13.1%。MTP2的单流优于MTP1，但四并发总吞吐略低于MTP1，差异小于MTP相对无MTP的收益；当前配置保留MTP2的原因是单流与短输出场景收益更高，若业务以四并发长输出为主，MTP1值得另行评估。

最终镜像MTP0相对a1旧profile的服务端Decode仅单流约+2.6%、四并发约+7.7%，不能把最终镜像相对a1的全部Decode差异归给WY。最终镜像四并发端到端吞吐约+20.1%，主要来自Prefill：同一4K请求的服务端Prefill约2175 vs 1296 tok/s（+67.7%），四并发请求阶段约1230 vs 516 tok/s（+138%）。同时最终配置使用主Graph捕获[1,2,4]（MTP0；MTP2为[3,6,12]），旧profile为[1,2]；batch tokens为2048 vs 4096；旧profile启用prefix cache但本次唯一salt命中为0。Graph档位、batch、prefix-cache开关以及最终镜像中的状态索引、slot int64和SplitFuse兼容修复均是整栈差异，不能单独拆成WY贡献。

旧profile不是a1实机压测，而是在a2用归档a1镜像、同一W8权重和旧启动参数复现；a1只读核对确认现役容器实际使用该镜像及参数：无MTP、Graph[1,2]、max-num-batched-tokens4096、prefix caching开启。完整结构化结果见`results/wy-decode-ablation-20261108/summary.json`。

## 部署资产与回滚

- `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-production.manifest.json`保留a1模型目录、18081端口及服务名，调用镜像内入口。
- `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-production.container`是a1路径的Quadlet模板，包含两卡设备、60秒冷却、失败重启和现有manifest launcher；显式覆盖ENTRYPOINT执行launcher。
- `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-image.env.example`用于直接调用镜像默认入口。

a1/a2驱动分别26.0.rc1和26.1.1；工程验收在a2完成，未自动在a1验证或部署。a1现有Quadlet由llmctl生成，部署时需同步其部署描述中的镜像与启动配置，避免再次渲染恢复旧值。切换前保存当前manifest、Quadlet与旧镜像digest；停止服务、确认NPU空闲、冷却至少60秒后启动。回滚同样执行冷却。不能只换manifest而继续使用旧镜像，也不能把旧Exec直接附加到新镜像ENTRYPOINT后。

结构化结果：`results/wy-mtp-production-20261108/main-graph-release-full-summary.json`。原始窗口保存在a2 runtime；`main-graph-release-packaged`实际438.875秒、driver0、清场通过。无源码挂载、关闭审计的二/三/四请求JSON、4K/32K、decode、混合请求及四并发PDF均通过；同image ID、同入口SHA、实际CLI已核对。registry manifest与config摘要也与该image ID一致，临时发布凭据已删除。完整汇总为`main-graph-release-summary.json`，发布身份见`graph-localize/main-graph-release-registry/registry-image.json`。

## a1 重建镜像 canary（2026-11-08）

失败 staging 镜像为 `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:777611b2e13da9be433e2e98ce67777713d07fc3321717af16a414e2594a5189`，远端 image ID 为 `bb033d6b18c6f6ee342f13b1457f5c8cdfeeb90698b3e68fa16f5535db952`；入口脚本虽匹配仓库，完整制品不匹配原验收清单。完整验收制品 `sha256:a11b8aca…` 已在 a1 以 prefix off 和 prefix on 各完成一次合成单请求 canary，均返回 `OK`，无507015/AICore/watchdog错误；a1随后恢复原服务。

候选于 21:35:32 +08:00 启动并通过 `/v1/models` 健康检查，但首次真实推理触发 NPU `507015`、HCCL watchdog 和 `EngineDeadError`，无有效速度结果，未通知前端切换。随后发现原有 a1 备份的 Quadlet 与 manifest 还存在旧镜像/入口不一致，恢复文件名并不能直接恢复 backend；该现场配置问题也已保留在 a1 journal。当前候选仅保留为 staging 诊断对象，不能作为生产或联调版本。


## a1 完整制品维护复验与回滚（2026-11-08）

恢复原完整制品 `a11b8aca…` 后，通过候选 Quadlet/manifest 临时部署，核验 build-manifest 中全部 **43** 个文件 SHA；实际 TP2/262K/seq4/batch2048/MTP2/主 Graph[3,6,12]/draft eager，prefix cache 开启。短 JSON、四并发 JSON、PDF 后 JSON 共6请求通过；两页 PDF 栅格 OCR 请求均以 stop 完成。32K+32冷请求完整返回，TTFT21.75秒、端到端22.50秒，prefix queries32768、hits0。候选窗口日志无507015/507903/AICore异常/HCCL watchdog异常/EngineDeadError。

同机先候选、后恢复原服务，每组4K输入/512输出，C1/C4各两轮，每请求独立salt，各自先预热32输出token；两组各10个计时请求全部成功。以下为HTTP客户端流式计时中位数，端到端吞吐包含Prefill，TPOT不是服务端纯Decode计数器。

| 配置 | 单路端到端 tok/s | 单路 TTFT 秒 | 单路 TPOT 毫秒 | C4聚合端到端 tok/s | C4 TTFT 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原服务1a193…，无MTP | 27.23 | 3.15 | 30.63 | 51.76 | 10.95 |
| 完整候选a11b8…，MTP2 | 35.56 | 2.19 | 23.90 | 59.32 | 6.45 |

本轮端到端吞吐单路约+30.6%、C4约+14.6%；只有两轮且候选C4两轮为66.41/52.23 tok/s，不能作为稳定百分比承诺或独立统计非劣证明。整栈/配置比较不能将全部收益归因于WY；本轮未重跑262K容量或正式OCR质量门禁。

测试后冷却60秒，恢复本轮备份的原Quadlet及manifest并逐字节核对一致；23:33:42 +08:00确认原镜像1a193…运行、systemd active、18081/v1/models=200，同样10个测速请求全部成功。候选未永久切换，n2未操作，未通知前端。结构化结果和原始证据SHA见 `results/wy-mtp-a1-maintenance-20261108/summary.json`。后续永久部署仍须同步llmctl站点/catalog/runtime配置源，避免重新渲染覆盖候选；本轮临时维护测试未修改该配置源。
