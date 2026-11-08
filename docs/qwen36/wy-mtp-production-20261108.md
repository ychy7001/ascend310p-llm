# TP2 W8A8 + WY + MTP2 满上下文部署配置

2026-11-08，a2完整容量、组合工程验收与同镜像内置入口验收均已通过，独立版本镜像已发布并完成registry manifest/config摘要核验。部署模板固定digest；a1未切换。本页不宣称正式OCR业务非劣效。

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
| prefix caching / async | 关闭 / 开启 |
| served model name | Qwen3.6-35B-A3B-w8a8 |

候选image ID：`bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4`。不可变镜像引用：`registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5`。registry验证时间为2026-11-08 06:56:19 UTC。

镜像元数据版本为torch2.10.0+cpu、torch_npu2.10.0.post2、vLLM0.23.0+empty、vllm-ascend0.23.0rc2.dev93+g88eac2713，CANN9.1.0-beta.1；本地overlay另外登记SHA，不把包版本当作未修改源码证明。

## Graph与状态修复

主模型图优化已开启。draft eager来自性能筛查：主图三档/draft eager约46 tok/s；双图受event资源限制需收敛到[12]，小批次padding后连续两窗约29 tok/s，Prefill相同。该筛查不是交替A/B，不据此宣称正式百分比收益。定位和无效实验标签见[Graph定位报告](wy-mtp-graph-localization-20261108.md)。

镜像包含PR18730配套recurrent状态表修复（经源码SHA核对复用opp-review构建，eager/Graph各42项独立回归通过），并将MTP slot block IDs在乘法前提升到int64，绕过模型上下文中的int32 Mul失败调用点。WY kernel、冻结协议/规则/worker/R和CANN版本保持不变。recurrent配套OPP、wrapper和proposer的变更独立登记，不能只替换Python wrapper。

## 完整窗口实测

`main-graph-release-full`耗时1324.247秒，driver0，独立清场通过。请求阶段计数器统计，三次正式请求取中位数；文本每请求独立cache salt，prefix hits为0。数字是服务Prefill阶段吞吐，不是纯kernel吞吐。

| 冷文本输入 | Prefill tok/s | TTFT秒 | 有效次数 |
| --- | ---: | ---: | ---: |
| 4K | 2049.6 | 2.002 | 3/3 |
| 8K | 1937.9 | 4.233 | 3/3 |
| 16K | 1746.8 | 9.390 | 3/3 |
| 32K | 1456.7 | 22.512 | 3/3 |

- 4K输入、128输出：decode中位数45.877 tok/s，draft acceptance约85.1%。
- 四并发端到端测试：12/12完成；含Prefill的聚合输出吞吐中位数35.703 tok/s，TTFT P50/P95为6.755/9.316秒。P95是12样本经验分位数。
- 二/三/四请求确定JSON通过，混合输入/输出长度4/4完成。
- 131072+128完整返回：TTFT177.967秒，Prefill736.752 tok/s。
- 262016+128完整返回，总token262144：TTFT595.558秒，Prefill440.041 tok/s；超限请求返回400。
- 四页PDF功能检查4/4；pdf-01达到512输出上限，其余stop。长请求后JSON通过。
- 每10秒npu-smi采样，观察到的单卡最高使用41760MB；不是allocator peak。

长输入测试证明容量和执行完整性，不证明262K检索语义质量。PDF仅功能检查；多模态处理缓存未专门禁用，不称为完全冷视觉吞吐。没有通过这些测试回溯修改旧G1结果，或新增正式OCR非劣效声明。

`max_num_seqs=4`是调度上限，不承诺四条262K请求同时驻留。启动报告KV pool约635026 tokens，即约2.42条满上下文请求，超出驻留容量由调度器处理。

## 部署资产与回滚

- `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-production.manifest.json`保留a1模型目录、18081端口及服务名，调用镜像内入口。
- `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-production.container`是a1路径的Quadlet模板，包含两卡设备、60秒冷却、失败重启和现有manifest launcher；显式覆盖ENTRYPOINT执行launcher。
- `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-image.env.example`用于直接调用镜像默认入口。

a1/a2驱动分别26.0.rc1和26.1.1；工程验收在a2完成，未自动在a1验证或部署。a1现有Quadlet由llmctl生成，部署时需同步其部署描述中的镜像与启动配置，避免再次渲染恢复旧值。切换前保存当前manifest、Quadlet与旧镜像digest；停止服务、确认NPU空闲、冷却至少60秒后启动。回滚同样执行冷却。不能只换manifest而继续使用旧镜像，也不能把旧Exec直接附加到新镜像ENTRYPOINT后。

结构化结果：`results/wy-mtp-production-20261108/main-graph-release-full-summary.json`。原始窗口保存在a2 runtime；`main-graph-release-packaged`实际438.875秒、driver0、清场通过。无源码挂载、关闭审计的二/三/四请求JSON、4K/32K、decode、混合请求及四并发PDF均通过；同image ID、同入口SHA、实际CLI已核对。registry manifest与config摘要也与该image ID一致，临时发布凭据已删除。完整汇总为`main-graph-release-summary.json`，发布身份见`graph-localize/main-graph-release-registry/registry-image.json`。
