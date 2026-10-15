# a2 后端接入与运维交付

## 交付组合

- 代码：https://github.com/ychy7001/ascend310p-qwen36 ，a2当前运行目录 `/home/ychy/services/qwen36/c16-20261015/repo`（固定提交5322528）；制品/参数快照见 `/home/ychy/services/qwen36/deployment.json`。旧repo保留，第二芯片继续使用原路径。
- 权重：https://modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P 。本机路径见 `deploy/a2.env` 的MODEL_DIR，不需重新下载。
- 服务：a2 TCP18024，模型名 `qwen36-w4a8-c8`。Tailscale：`http://100.92.170.57:18024/v1`；同局域网：`http://192.168.3.133:18024/v1`。客户端需能连通相应网络。
- 用户systemd：`qwen36-a2.service`；rootless Podman容器 `qwen36-tp1-48g-112k`，只使用davinci0。芯片1保留原W4对照，18028不是业务入口。

固定组合为TP1/48G、W4A8、C16（FP16 KV）、MTP2、Decode Graph、GDN、MindIEQ E5，4096/512，最大总序列114688 token，显存利用率0.95。单请求调度；多客户端会排队，不代表已支持并发批处理。请求输入与输出预算合计需≤114688。prefix默认关闭，图像输入未验收。

2026-10-15更新：公开候选入口优先C16；测试结果与切换记录见[a2 C16/C8配对报告](../results/a2-kv-comparison-20261015/README.md)。API模型名保留`qwen36-w4a8-c8`兼容已有客户端，不能据名称推断KV格式。显式`QWEN36_C8_ENABLED=1`可重建C8。独立实验前先停systemd管理单元，避免它按旧参数自动拉起容器；不要同时管理同一设备/端口。当前实际容器与管理状态以该报告最后快照为准。

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
systemctl --user status qwen36-a2.service
journalctl --user -u qwen36-a2.service -n 100 --no-pager
systemctl --user restart qwen36-a2.service
curl --fail http://127.0.0.1:18024/health
```

重启需要重新加载模型和捕获Graph；systemd running不代表模型ready，以HTTP健康检查200为准。用户已开启linger，服务enable后可随用户manager启动。停止用 `systemctl --user stop qwen36-a2.service`，不要同时执行旧实验启动命令占用同一芯片。

变更前校验及制品追溯：

```bash
cd /home/ychy/services/qwen36/c16-20261015/repo
source deploy/a2.env
bash scripts/run-candidate.sh --check
cat /home/ychy/services/qwen36/deployment.json
```

`deployment.json`记录部署提交、镜像与制品SHA256；源码和二进制分别追溯，不宣称本次重建了全部kernel。权重保持原发布文件，未覆盖或重新量化。升级需另建指向审查后提交的worktree，校验制品后更新systemd工作目录/启动路径，再重启。当前运行目录是detached worktree，不直接git pull，也不自动追随main。

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
