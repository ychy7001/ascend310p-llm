# a2 后端接入与运维交付

## 交付组合

2026-10-16 起，a2 默认交付入口为已验收的 digest 固定运行镜像和 `qwen36-image.service`。旧的 `qwen36-a2.service`、源码 worktree、OPP 和动态库仍保留，仅用于回滚和对照；不要同时启用两个 unit。镜像发布与切换证据见 [`results/runtime-image-20261016/`](../results/runtime-image-20261016/)，运行代码、OPP 和扩展库均来自镜像内的 168 文件哈希清单，权重没有重新下载或覆盖。

- 代码：https://github.com/ychy7001/ascend310p-qwen36 。镜像运行源码固定提交 `c9fd1da`；主机配置为 `~/.config/qwen36-image.env`，仓库快照见 [deploy/a2-image-20261016.env](../deploy/a2-image-20261016.env)。旧源码目录 `/home/ychy/services/qwen36/c16-20261015/repo`（5322528）仅作回滚保留，第二芯片仍用原路径。
- 权重：https://modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P 。本机路径见 `deploy/a2-image-20261016.env` 的 MODEL_DIR，不需重新下载。
- 服务：a2 TCP18024，模型名 `qwen36-w4a8-c8`。Tailscale：`http://100.92.170.57:18024/v1`；同局域网：`http://192.168.3.133:18024/v1`。客户端需能连通相应网络。
- 用户systemd：`qwen36-image.service`；rootless Podman容器 `qwen36-image-managed`，只使用davinci0。芯片1保留原W4对照，18028不是业务入口。

固定组合为TP1/48G、W4A8、C16（FP16 KV）、MTP2、Decode Graph、GDN、MindIEQ E5，2048/512，最大总序列114688 token，显存利用率0.95。单请求调度；多客户端会排队，不代表已支持并发批处理。请求输入与输出预算合计需≤114688。prefix默认关闭，图像输入未验收。

2026-10-15更新：公开候选入口优先C16；测试结果与切换记录见[a2 C16/C8配对报告](../results/a2-kv-comparison-20261015/README.md)。API模型名保留`qwen36-w4a8-c8`兼容已有客户端，不能据名称推断KV格式。显式`QWEN36_C8_ENABLED=1`可重建C8。独立实验前先停systemd管理单元，避免它按旧参数自动拉起容器；不要同时管理同一设备/端口。当前镜像与管理状态以[镜像切换记录](../results/runtime-image-20261016/README.md)为准；2048部署快照和旧 deployment.json 是切换前的历史记录。

## MTP与Prefill batch选择依据

[a2四组参数对照](../results/a2-profile-matrix-20261015/README.md)已完成16个正式请求。32K同输入/同输出/接受率100%下，MTP2比MTP1提高Decode约23%，Prefill下降不足1%；固定MTP后2048与4096速度接近。用户授权后实际尝试三台MTP2/2048/512；a2通过systemd dropin设置2048，并完成本轮32K、112000+128、112K会议及长后正确性门禁。32K中位Prefill/Decode为550.15/32.50 tok/s，112000/128为384.35/21.70 tok/s。clt与RC的MTP2出现可复现故障，当前恢复MTP1，不照搬a2结论；[原始验收与失败证据](../results/unified-2048-20261015/README.md)独立保留。业务接受率和质量、structured output边界仍须按下文处理；此对照不扩展任何生产承诺。

## HTTP调用

```bash
curl --fail http://100.92.170.57:18024/health
curl --fail http://100.92.170.57:18024/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-w4a8-c8","messages":[{"role":"user","content":"请简要说明会议纪要应包含哪些内容。"}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

支持OpenAI兼容Chat/Completions与stream。服务自身未配置API认证，由后端在受控网络接入。建议显式设置输出预算、请求取消及足够的超时；112K冷输入首文本可能需约5分钟，不能套用短对话超时。服务端max_model_len是输入+输出总预算，不是单次输入可用长度。

## 重启、检查与日志

以下命令以 `ssh ychy-a2` 后的ychy用户运行：

```bash
systemctl --user status qwen36-image.service
journalctl --user -u qwen36-image.service -n 100 --no-pager
systemctl --user restart qwen36-image.service
curl --fail http://127.0.0.1:18024/health
```

重启需要重新加载模型和捕获Graph；systemd running不代表模型ready，以HTTP健康检查200为准。用户已开启linger，服务enable后可随用户manager启动。停止用 `systemctl --user stop qwen36-image.service`，不要同时执行旧实验启动命令占用同一芯片。

镜像与有效配置追溯（不初始化 NPU）：

```bash
# 此文件只含部署参数，不放认证信息。
source ~/.config/qwen36-image.env
podman run --rm --entrypoint python3 "$IMAGE" -S /opt/qwen36/verify-image.py --base
podman exec qwen36-image-managed cat /opt/qwen36/manifest.json
```

镜像源码提交与文档提交分别追溯；仅整理文档不会自动重建镜像。更新部署参数后需 `systemctl --user restart qwen36-image.service` 重建容器，直接 `podman start` 不能更新旧容器环境。升级应先在隔离候选通过门禁，再修改配置中的 IMAGE digest；不追随可变 tag 或 main。

回滚到保留的源服务：

```bash
systemctl --user disable --now qwen36-image.service
systemctl --user enable --now qwen36-a2.service
curl --fail http://127.0.0.1:18024/health
```

旧源码与制品只作回滚，不需要后端直接 git pull；权重保持原发布文件，未覆盖或重新量化。

## 后端验收与边界

已有固定token长上下文吞吐、needle、64题子集及112K跨段决议更新测试，原始结果保留于results。历史C8近112K重复合成输入约379 Prefill/16.9 Decode tok/s；历史C8中文决议更新canary约345/16.5 tok/s。当前C16测试以2026-10-15配对报告为准。单个合成canary不替代业务质量，实际内容影响路由和速度。

后端接手真实会议摘要、待办/负责人/日期提取质量、长输出、断连取消、排队、超时与prefix增量复用评估。当前交付是可接入实验候选，尚未宣称并发P95、TP2 C8、32G或多模态生产验收。

## 310P GDN + MTP 结构化输出风险（Issue #17856）

当前 a2 运行时使用的 `recurrent_gated_delta_rule_v310` OPP 源码仍包含如下边界路径：当上一轮接受的 token 数量 `acceptedTokenNum` 大于当前请求的 `seqLen` 时，在 kernel 内直接 `return`。这与上游 [Issue #17856](https://github.com/vllm-project/vllm-ascend/issues/17856) 描述的 310P + GDN + MTP + structured output 问题一致。它属于正确性风险，可能表现为结构化结果截断、缺失或状态未写入，不一定产生 Python 异常。

本交付服务设置 `max_num_seqs=1`，降低了多请求变长 batch 触发该路径的概率，但结构化输出/grammar 仍可能改变每轮实际 query 长度，因此不能宣称已经规避。此前的 8K、32K、112K canary 是普通文本提示，未使用 `response_format` 或 grammar，不能作为该问题的回归测试。后端若需要严格 JSON/Schema，建议在专门修复并重建 310P OPP 前使用 MTP=0/1 的隔离服务，或暂不把该能力作为产品承诺；不要仅删除 `return`，因为状态索引协议仍需验证。

当前普通文本的接受率没有显示出该 bug 的直接证据：a2 最近两次服务窗口记录约 92.0% 和 94.9%，且无错误日志。vLLM 同时会对 `num_speculative_tokens > 1` 给出接受率可能下降的提示；因此此前 MTP 接受率变化应先按 MTP1/MTP2、同一输入输出预算、无 grammar 的交替测试归因，不能直接归因于 #17856。

修复方向应参考 [PR #17839](https://github.com/vllm-project/vllm-ascend/pull/17839) 的状态槽/逐请求边界思想，但不能直接把 910 系列改动套到 310P。下一步应先做不依赖完整模型的 310P 单算子复现：构造当前长度 `[2,4]`、上一轮接受数 `[4,1]`，用可区分状态值检查是否出现提前退出或跨请求状态；之后再修改 v310 kernel、Python flatten/index 协议并重建 OPP，最后做 structured-output、MTP1/2 和 Graph 回归。当前服务不因该 Issue 自动改 kernel。

## 本次代码收束

默认入口直接调用容器启动器；删去旧Prefill/MTP/Graph/长上下文批量sweep工具（历史版本在3ce8cb9）。保持已测运行适配层与kernel源码，避免交付前无收益地重写底层。保留构建、制品校验、单算子测试、HTTP验收工具和原始结果以便定位问题。历史文档中的已删脚本属于当时实验记录，应在相应提交复现。

## 2026-10-16 EP复验与部署边界

clt原MTP1已完成同协议32K/112K/会议与长后质量，112000/128为Prefill417.49、Decode16.61 tok/s；MTP2独立修复候选仍5/8，未做性能测试。[本轮原始记录](../results/ep-convergence-20261016/README.md)保留失败输出和恢复验收。交付继续使用a2 MTP2/2048、clt MTP1/2048、RC MTP1/4096，不将未通过的修复接入默认代码。两个EP当前prefix均关闭，a2加载视觉而clt为language-model-only；这些是当前事实，不是多模态或prefix业务验收。公共参数与例外统一以[Profile说明](profiles-20261014.md)为准。
