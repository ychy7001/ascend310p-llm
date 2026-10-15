# clt Qwen3.6-35B-A3B W4A8 部署交接

更新：2026-10-15（北京时间）。本文供同事**直接登录clt主机**操作；以下命令均在clt执行，无需ychy-a2跳板。当前服务已经部署，通常从第2节开始，不需要下载权重、编译算子或重建环境。

## 1. 当前交付配置

- 主机：`192.168.29.49`，用户`clt`，Ubuntu24.04/x86_64、PCIe EP、单颗Ascend310P1。
- 设备：`/dev/davinci0`，npu-smi id49680，设备可见总内存44215MB。标称48G，不等于48GiB可分配空间。
- 当前容器：`qwen36-c16-112k-20261015`；接口`http://127.0.0.1:18016/v1`，局域网地址`http://192.168.29.49:18016/v1`。
- OpenAI API模型名：`qwen36-w4a8-c8`。这是兼容旧客户端的名字；**当前实际KV存储为C16/FP16**，不要从名字判断。
- 默认：TP1 / W4A8 / C16 / MTP=1 / Decode Graph / GDN与MindIEQ优化；同步调度、仅文本、Prefix Cache关闭、最大同时请求数1。
- `max_model_len=114688`（输入+输出总长），已验证112000输入+128输出；不是硬件最大上下文规格。
- `gpu_memory_utilization=0.90`、`max_num_batched_tokens=2048`、`max_num_seqs=1`。
- Graph：`ENFORCE_EAGER=0`、`COMPILATION_MODE=0`、`FULL_DECODE_ONLY`、capture sizes`1,2`。不是Prefill全图捕获。
- Docker镜像：`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`；已验证镜像ID `sha256:16770d322d0e343bc5ff61f897acfc6fb1383eee56d7939554e1c9cf8b504773`。
- 已记录的软件栈：npu-smi26.0.t1、CANN9.1、torch2.10、torch-npu2.10.post4。复用本机Docker；没有迁移Podman或升级驱动。
- `clt`已加入`docker`和`HwHiAiUser`。其他账号需具备相应权限；新加组后重新登录。

## 2. 资源路径

| 用途 | clt主机绝对路径 |
|---|---|
| 本交接资料 | `/home/clt/qwen36-handoff-20261015/` |
| 默认部署根目录 | `/home/clt/qwen36-optimized-20261015/` |
| 默认启动入口 | `/home/clt/qwen36-optimized-20261015/start-candidate.sh` |
| 当前参数 | `/home/clt/qwen36-optimized-20261015/candidate.env` |
| 当前身份与状态说明 | 上述目录的`current-default.json`、`runtime-identity.json` |
| 原始W4混合权重，模型/MTP索引 | `/home/clt/qwen36-w4-deploy/model/` |
| 冻结运行时源码副本 | `/home/clt/qwen36-optimized-20261015/repo/` |
| Docker创建脚本 | `/home/clt/qwen36-optimized-20261015/repo/experimental/run-c8-service.sh` |
| 算子制品目录 | `/home/clt/qwen36-optimized-20261015/artifacts/` |
| W4自定义OPP与torch扩展 | 上述目录的`w4-opp/`、`qwen36_int4_custom.so` |
| GDN OPP、扩展、Python实现 | 上述目录的`gdn-opp/`、`pr17640_gdn.so`、`gdn-source.py` |
| MindIEQ Prefill扩展 | 上述目录的`libqwen36_mindieq_grouped.so` |
| C8扩展，当前关闭但启动器仍挂载 | 上述目录的`libqwen36_c8.so` |
| clt同步元数据适配 | `/home/clt/qwen36-optimized-20261015/repo/experimental/clt_sync_metadata.py` |
| KV分配审计 | 同目录的`clt_kv_audit.py`，启动时`CLT_KV_AUDIT=1` |
| bootstrap | `/home/clt/qwen36-optimized-20261015/repo/experimental/c8_bootstrap/sitecustomize.py` |
| HTTP测速工具 | `/home/clt/qwen36-optimized-20261015/repo/tools/` |
| 完整验证脚本 | `/home/clt/qwen36-optimized-20261015/validate_deployment.py` |
| C16/C8对照与当前切换记录 | `/home/clt/qwen36-kv-comparison-20261015/` |
| C16 8K/32K/112K原始测试 | 上述目录的`c16/logs/` |
| 当前C16重启验证 | 上述目录的`c16-default-sanity.json`、`c16-default-8192.jsonl` |
| 旧C8结果，作为历史对照 | `/home/clt/qwen36-optimized-20261015/logs/`及`summary.json` |
| 默认切换前参数备份 | `/home/clt/qwen36-optimized-20261015/candidate.env.before-c16-default` |

`repo/`是冻结的运行时副本，不是可直接git pull升级的工作树。容器把其中runtime、bootstrap、experimental和制品只读挂载；不要一边服务运行一边修改挂载源码。默认目录的`summary.json`和`DEPLOYMENT_RECORD_COMMIT`来自首次C8部署，不能当作当前C16状态；当前以`candidate.env`、`current-default.json`及容器inspect为准。

公开代码：[GitHub](https://github.com/ychy7001/ascend310p-qwen36)。公开权重：[ModelScope](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P)。本机权重已完成59文件校验，无需重新下载；不要覆盖或移动其中被索引引用的分片。

## 3. 检查与启动现有服务

```bash
id
npu-smi info
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
cat /home/clt/qwen36-optimized-20261015/current-default.json
cat /home/clt/qwen36-optimized-20261015/candidate.env
curl --fail --max-time 10 http://127.0.0.1:18016/health
curl --fail --max-time 10 http://127.0.0.1:18016/v1/models
```

已经健康运行就直接使用。容器已存在但停止时，确认芯片没有另一套35B服务，再运行：

```bash
bash /home/clt/qwen36-optimized-20261015/start-candidate.sh
docker logs --tail 100 -f qwen36-c16-112k-20261015
```

Ctrl+C只结束日志跟随，不停止容器。`docker start`返回不代表模型已加载；本次启动通常需要几分钟，以health HTTP200为准。容器设置`unless-stopped`；已配置自动重启策略，但未做宿主机重启验收。

```bash
docker inspect --format '{{.Name}} running={{.State.Running}} restart_count={{.RestartCount}} restart={{.HostConfig.RestartPolicy.Name}}' qwen36-c16-112k-20261015
```

停止：`docker stop qwen36-c16-112k-20261015`。重新加载原容器配置：先stop再执行默认启动入口。**编辑candidate.env之后docker start不会更新既有容器的环境变量和命令行。** 参数变化使用第6节的独立容器流程。

## 4. API调用与快速验证

```bash
curl --fail --max-time 120 http://127.0.0.1:18016/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-w4a8-c8","messages":[{"role":"user","content":"计算37乘19，只回答数字。"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

预期答案703。当前接口没有API key配置，`--language-model-only`，不接受视觉能力验收。API调用者设置输入/输出总长不超过114688；112K冷输入本次TTFT约4.7分钟，长请求客户端需允许至少600秒，完整长测流程约15分钟或更久。

测试时独占端点，不要同时从其他客户端发请求，否则阶段指标差值会混入别的请求。以下创建独立结果目录，不覆盖历史结果：

```bash
CLT_TEST_DIR=$(mktemp -d /home/clt/qwen36-check.XXXXXXXX)
printf '%s\n' "$CLT_TEST_DIR"
python3 /home/clt/qwen36-optimized-20261015/repo/tools/bench_generation_sanity.py \
  --base-url http://127.0.0.1:18016 --model qwen36-w4a8-c8 \
  --out "$CLT_TEST_DIR/sanity.json" --max-tokens 64
python3 /home/clt/qwen36-optimized-20261015/repo/tools/bench_server_phases.py \
  --base-url http://127.0.0.1:18016 --model qwen36-w4a8-c8 \
  --out "$CLT_TEST_DIR/p8192.jsonl" --prompt-tokens 8192 \
  --output-tokens 128 --repeats 3 --unique-prompts --spec-metrics
npu-smi info > "$CLT_TEST_DIR/memory-after.txt"
```

五类sanity只是检查空输出和明显退化，不等于正式精度通过。bench包含预热，表格用三轮request中位数；Prefill/Decode是服务阶段指标，不等于输出token除HTTP总耗时。

## 5. 完整复现8K/32K/112K检索与短请求回归

保持当前C16参数，独占服务端点。无需停止或重新创建容器。

```bash
CLT_TEST_DIR=$(mktemp -d /home/clt/qwen36-full-check.XXXXXXXX)
mkdir -p "$CLT_TEST_DIR/logs"
ln -s /home/clt/qwen36-optimized-20261015/repo "$CLT_TEST_DIR/repo"
nohup python3 -u /home/clt/qwen36-optimized-20261015/validate_deployment.py \
  --deployment "$CLT_TEST_DIR" > "$CLT_TEST_DIR/worker.log" 2>&1 < /dev/null &
printf '测试PID=%s，结果目录=%s\n' "$!" "$CLT_TEST_DIR"
tail -f "$CLT_TEST_DIR/worker.log"
```

记录结果目录；断开SSH后可用实际目录替换变量重新查看。完成条件：logs/validation-progress.jsonl最后一行`stage=complete`且`pass=true`，112K early/late均`valid_metrics=true`、`answer_starts_with_expected=true`，固定题9/9。

```bash
cat "$CLT_TEST_DIR/logs/validation-progress.jsonl"
cat "$CLT_TEST_DIR/logs/112k-post-long-fixed-answers.json"
```

112K测试为两个位置各一次，没有同长度单独warmup；强制ignore_eos输出128token用于测速，只校验首段检索答案，不能把后续重复模板视作自然对话质量。`memory-*.txt`是阶段结束快照，不是全程峰值。如评估最高峰值，可另外采样npu-smi并记录时间，但10秒采样也可能漏瞬态峰值。

## 6. 修改参数后独立复现容器（按需操作）

只有修改配置或需要验证容器创建流程时才执行。保留默认容器作为回退，不删除旧容器或权重。先查看`docker ps`和`npu-smi info`，停止当前同芯片模型；不要执行其他历史目录里的run.sh。

```bash
CLT_REPRO_DIR=$(mktemp -d /home/clt/qwen36-repro.XXXXXXXX)
cp /home/clt/qwen36-optimized-20261015/candidate.env "$CLT_REPRO_DIR/candidate.env"
# 按需编辑副本；先保持原配置可复现，再逐项改参数。
${EDITOR:-vi} "$CLT_REPRO_DIR/candidate.env"
docker stop qwen36-c16-112k-20261015
(
  set -e
  source "$CLT_REPRO_DIR/candidate.env"
  export CONTAINER_NAME="qwen36-repro-$(date +%Y%m%d-%H%M%S)"
  printf '%s\n' "$CONTAINER_NAME" > "$CLT_REPRO_DIR/container-name.txt"
  bash /home/clt/qwen36-optimized-20261015/repo/experimental/run-c8-service.sh
)
CLT_REPRO_CONTAINER=$(cat "$CLT_REPRO_DIR/container-name.txt")
docker logs --tail 100 -f "$CLT_REPRO_CONTAINER"
```

这个流程沿用18016和已有制品，务必停旧服务。health通过后执行第4/5节；如果失败，记录日志、停止新容器，再启动默认容器：

```bash
docker logs "$CLT_REPRO_CONTAINER" > "$CLT_REPRO_DIR/container.log" 2>&1
docker stop "$CLT_REPRO_CONTAINER"
bash /home/clt/qwen36-optimized-20261015/start-candidate.sh
```

创建脚本名含c8不表示启用了C8；取决于`QWEN36_C8_ENABLED`。C16默认为0。当前镜像内已有基础vLLM/Ascend包，单独从官方main安装并不能自动得到这些W4/GDN/MindIEQ补丁。复现必须同时保留冻结源码、clt bootstrap/同步元数据补丁、OPP、.so和权重。

## 7. C8备选、恢复默认及排障

C8仅作为对照/内存受限备选，切换会中断当前请求：

```bash
docker stop qwen36-c16-112k-20261015
docker start qwen36-optimized-112k-20261015
# 观察加载及health，API模型名/端口不变
```

恢复默认C16：

```bash
docker stop qwen36-optimized-112k-20261015
bash /home/clt/qwen36-optimized-20261015/start-candidate.sh
```

临时切C8不会修改candidate.env，因此下次默认入口仍启动C16。两套35B不能同时加载到同颗NPU。旧`qwen36-w4a8-tp1`/18015是eager/MTP0历史服务，不是当前默认或同条件基线。

- `aclInit 507899`/runtime初始化失败：先核查npu-smi进程和其他同芯片容器，不能直接判成算子数值失败。
- OPP/.so加载失败：核对candidate.env指向的文件与Docker挂载、镜像ID、x86_64架构；不要套用OrangePi ARM64库。
- 改.env后效果没变：inspect既有容器环境/Args；`docker start`不读取新.env。
- 看见KV capacity241277不代表已验收241K上下文；当前服务上限与真实测试长度见第1节。
- 内存与速度异常：保证无其他NPU负载、Prefix off、单流、相同输入/输出；保留docker logs、npu-smi和bench原始JSONL。
- 设备已有`Warning A419321C lp pmbus error`，软件门禁通过不表示硬件告警已解决。本文不授权驱动重装或主机升级。

## 8. 来源与适配范围

冻结核心源码基于仓库提交`e3fc3f9e85d7cd372d8a225317f24abea6f67861`，clt额外应用Docker启动器、bootstrap、clt_sync_metadata以及仅启动期的KV审计模块。制品复用a2现成版本，并在本机310P1实测；不是声称官方main已包含所有功能。

- 初次clt部署记录：`results/clt-deployment-20261015`，提交`2fdbf975824e5656f9c7a25083e5d0e37ba29dd5`。
- C16/C8配对：`results/clt-kv-comparison-20261015`，提交`e1b0c66`。
- C16默认切换和预算：`results/clt-context-selection-20261015`，提交`9eb87f8`。
- 本交接目录中的`current.env`、`start-current.sh`、`runtime-identity.json`及SHA256清单是交接时快照；实际启动仍使用默认部署根目录，不要从交接目录启动副本。

本指南覆盖clt既有资源复现，不是另一台空白机器从零安装指南。跨主机需要重新核对硬件、镜像、制品架构和驱动；不要只复制本文启动命令。
