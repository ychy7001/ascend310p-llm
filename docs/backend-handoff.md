# a2 后端接入与运维交付

## 交付组合

- 代码：https://github.com/ychy7001/ascend310p-qwen36 ，a2部署目录 `/home/ychy/services/qwen36/repo`。实际Git提交见同级 `deployment.json`。
- 权重：https://modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P 。本机路径见 `deploy/a2.env` 的MODEL_DIR，不需重新下载。
- 服务：a2 TCP18024，模型名 `qwen36-w4a8-c8`。Tailscale：`http://100.92.170.57:18024/v1`；同局域网：`http://192.168.3.133:18024/v1`。客户端需能连通相应网络。
- 用户systemd：`qwen36-a2.service`；rootless Podman容器 `qwen36-tp1-48g-112k`，只使用davinci0。芯片1保留原W4对照，18028不是业务入口。

固定组合为TP1/48G、W4A8、动态C8、MTP2、Decode Graph、GDN、MindIEQ E5，4096/512，最大总序列114688 token，显存利用率0.95。单请求调度；多客户端会排队，不代表已支持并发批处理。请求输入与输出预算合计需≤114688。prefix默认关闭，图像输入未验收。

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
cd /home/ychy/services/qwen36/repo
source deploy/a2.env
bash scripts/run-candidate.sh --check
cat ../deployment.json
```

`deployment.json`记录部署提交、镜像与制品SHA256；源码和二进制分别追溯，不宣称本次重建了全部kernel。权重保持原发布文件，未覆盖或重新量化。升级需显式拉取审查后的提交，重新校验再重启；运行目录不自动追随main。

## 后端验收与边界

已有固定token长上下文吞吐、needle、64题子集及112K跨段决议更新测试，原始结果保留于results。近112K重复合成输入约379 Prefill/16.9 Decode tok/s；中文决议更新canary约345/16.5 tok/s。单个合成canary不替代业务质量，实际内容影响路由和速度。

后端接手真实会议摘要、待办/负责人/日期提取质量、长输出、断连取消、排队、超时与prefix增量复用评估。当前交付是可接入实验候选，尚未宣称并发P95、TP2 C8、32G或多模态生产验收。

## 本次代码收束

默认入口直接调用容器启动器；删去旧Prefill/MTP/Graph/长上下文批量sweep工具（历史版本在3ce8cb9）。保持已测运行适配层与kernel源码，避免交付前无收益地重写底层。保留构建、制品校验、单算子测试、HTTP验收工具和原始结果以便定位问题。历史文档中的已删脚本属于当时实验记录，应在相应提交复现。
