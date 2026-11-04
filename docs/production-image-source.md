# 生产镜像源码化：第一阶段取证阻塞

2026-11-04：尚未完成源码重建。层与配置证据见
[`production-forensics.json`](../deploy/image/production-forensics.json)。其中 registry 使用变量；该文件是取证记录，不是可用的构建锁文件。

生产镜像为 `${PRODUCTION_REGISTRY}/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38`，amd64，共 22 层。历史显示前 19 层来自
`${UPSTREAM_REGISTRY}/ascend/vllm-ascend:nightly-releases-v0.23.0-310p-openeuler`，最后一个基础层构建于 2026-08-07。其后的三个文件系统层记录：

| 历史记录 | 来源 / 上游 PR 或 issue / 必要性 |
| --- | --- |
| 覆盖 `vllm_ascend/_310p/attention/attention_mask.py` | 生产层历史，标签 revision 为 `f8f58951e43054b2da7f7e200939d01f933197cb` / 无（尚未核实）/ 需提取文件差异后确认用途 |
| 覆盖 `tests/ut/_310p/attention/test_attention_mask_310.py` | 同上 / 无（尚未核实）/ 需与对应源文件一起恢复并核验 |
| `python3 -m pip install --no-deps triton==3.2.0` | 生产层历史 / 无（尚未核实）/ 重建需保留此依赖，但仍需取得 wheel 哈希和安装结果 |

以上不是完整文件差异清单，也尚未形成可应用补丁。镜像标签、版本参数和用户提供的短提交均不能代替镜像内 Git 取证。

## 原始基础镜像尚未取得

同名官方 tag 当前的 amd64 manifest 为 `sha256:5ab669ef238b64d2ba6e899d15ed02118f69558a248ebada1723a667892bd48b`，构建于 2026-08-11。其首层未压缩哈希与生产不同，不能用它替代历史基础。

官方 tag 历史接口列出 2026-08-07 06:47:42 UTC 发布的候选索引 `sha256:538a7c0c52bc0c117b27fbf5bc650b22ee2bb200014890b9999f2b33b2f901a6`。该索引及其两个子 manifest 的 registry 读取均返回 HTTP 404；历史接口的 presence 字段不能证明实际可拉取。由于配置未取得，不能确认这就是生产使用的索引，也不能确定两个子 manifest 的架构。

按任务的计划外问题停止规则，中止正在进行的生产镜像拉取，未启动额外的大型解包、哈希或构建。没有官方原始基底的文件树，无法完成要求的官方逐文件对照，或为候选给出已经核验的基础 digest。

继续所需材料是可读取、且前 19 层未压缩哈希与取证记录完全匹配的官方原始镜像。若要改为从生产镜像前 19 层恢复基底，须由 owner 确认调整范围；这种恢复不等于从官方 registry 独立取得原始基底，不能代替本阶段要求的官方对照。

## 与现有构建入口的关系

现有 `deploy/image/base-amd64.json` 的多架构 digest 为 `sha256:16770d322d0e343bc5ff61f897acfc6fb1383eee56d7939554e1c9cf8b504773`，amd64 子 manifest 为 `sha256:aa625af28af09b0a4d001dcd227820bd8e4c0ad023d3c33c64d36d4bc367fd73`，构建于 2026-09-22，共 18 层。与生产的首层未压缩哈希不同。

该锁文件声明 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、vllm-ascend `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。vLLM 短提交与用户提供的生产信息重合，但尚未在生产文件树核实；vllm-ascend 的生产 HEAD 未知。

`tools/prepare_runtime_image.py` 打包已测 W4/C8/GDN/MindIEQ 二进制和 bootstrap；`Containerfile` 验证文件后移动 triton；`start.sh` 固定 W4A8/TP1，设置自定义 OPP、扩展及 PYTHONPATH。生产配置则保留官方 entrypoint，并显式安装 triton。结论是 **另一条 W4/TP1 运行镜像线，部分上游来源重合**，不能直接用于 W8A8/TP2 生产复现。

现有 `patches/qwen36/gdn-17856-state-table.patch` 和 `patches/qwen38/` 的实验补丁没有生产已应用证据，不应自动加入生产队列。两条线有共同的容器构建、来源锁定与无设备校验需求，后续应扩展现有 `deploy/image`；尚无理由新建第二套机制。本轮保持现有构建文件和补丁不变，等待原始基底取证完成后再决定最小改动。

## 第二阶段验证计划（待 owner 审批，未执行）

前提是第一阶段已经完成：官方基底可读取、两份上游完整提交已核实、每项差异均有补丁或显式构建步骤、候选已在 CPU 构建，源码逐文件哈希、pip freeze、环境变量及入口对照完成。时间戳或缓存以外的差异必须逐项解释。

1. 在验证窗口冻结同一份 Qwen3.6-35B-A3B W8A8 权重及 tokenizer 哈希、生产实际启动命令和环境、驱动与设备映射。候选仅替换镜像，保持 TP2、dtype、MTP、Graph、prefix cache、模型长度、batch、seqs 和内存参数完全一致。归档 `experimental/runs-20261103-04/production-manifest.json` 可供核对，但不能未经确认就当成当时线上有效参数。
2. A 为生产 digest，B 为候选，A′ 为生产镜像的新进程；所有臂使用同一设备、相同冷启动和请求顺序，避免并行争抢设备。先采 A/A′ 冻结噪声阈值，再采 B；实验结束恢复原服务并独立核验 health、有效配置与短质量。A′可在 B 前采集用于冻结阈值，B 后另做恢复检查。
3. G1 沿用 `tools/compare_token_logprobs.py` 及 `results/wy921-gate-20261104/` 的协议：冻结同一组 token 输入，覆盖既有回放、4K 边界及 16K/32K 会议输入；greedy、seed 0、每条输出 256 token，采生成 top-20 logprob。使用已有共同显式支持条件 KL、Top-1 一致率、目标 token logprob 差和首分歧指标，阈值在 B 前冻结为 A′/A 实测误差的 1.5 倍，不沿用旧实验的具体数值。保留 top-k 删失下界，无法判定时不宣布通过；不把条件 KL 当完整词表 KL。不新增 prompt-logprob OOM 探测或改变生产内存参数。
4. G2 沿用 `tools/eval_gsm8k_http.py`、`tools/check_cache_request_reuse.py`、`tools/check_business_facts.py` 和 `tools/check_business_quality.py`。冻结同一 GSM64 的数据 SHA、indices、模板和输出预算；比较逐题答案、正确率、截断和解析缺失，执行短题前后测、请求复用/canary、结构化字段与事实检查，并人工检查会议负责人、期限和完成状态。保留生产既有错误，不以自动 schema 通过代替语义验收；候选不得新增回归。
5. 两项门禁通过后，以 `tools/bench_http_stream.py` 执行 1024/4096/16384/32768 输入 × C1/C4 的八格矩阵；所有臂固定 seed、输出 128 token 和相同预热，每格至少三轮有效测量，保留逐请求失败、usage、TTFT、TPOT、总耗时及吞吐。用 cache salt 并核实冷缓存命中为 0，保持服务 prefix 参数不变。对比 B/A、B/A′ 和 A′/A 漂移；要求成功率和质量不下降、吞吐不降低、延迟不增加，噪声范围及重复规则在看 B 前冻结。证据不足时不判通过，不要求额外性能收益。

本计划不授权部署、推送或修改下游引用；本轮没有执行设备验证。
