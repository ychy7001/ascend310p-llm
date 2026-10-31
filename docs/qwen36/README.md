# Ascend 310P — Qwen3.6-35B-A3B W4A8 / MTP / C8

在 Atlas 300I DUO（Ascend 310P3）单卡上跑 Qwen3.6-35B-A3B：routed experts 量化为 W4A8，叠加 BF16 MTP 推测解码、INT8 KV（C8）和 Decode Graph。W4A8 + MTP + C8 + Graph 已经在真实 310P3 上通过 4K / 32K / 112K 三档的启动、正确性和长上下文检索门禁。

Graph 路径用的是实验性 updatable-graph 开关，不是上游官方默认实现。本项目不是华为、Qwen 或 vLLM 官方发布。

## 当前状态

| 能力 | 状态 |
| --- | --- |
| W4A8，TP1/TP2，FP16 KV | 跑通 |
| BF16 MTP（1 枚草稿） | TP1/TP2 跑通 |
| BF16 MTP（2 枚草稿）+ C8 + Graph | 4K/32K/112K 全部通过 |
| INT8 KV（C8） | 已验证，仍是实验后端 |
| Decode Graph | 已在 310P 完成 capture/replay；正式 updatable-graph backend 修复待上游确认 |
| 长上下文 | 112K 近上限检索已通过；不等于 256K 验收 |
| TP2 C8、并发批处理、多模态、正式 P95 | 未验收 |

权重已发布：[ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P)（ModelScope）。发布校验细节见[发布校验记录](release-validation.md)。

## 快速开始

环境：Ascend 310P3、CANN 9.1、镜像 `quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`、Torch 2.10 + torch-npu 2.10.0.post4。SDK 和驱动由使用者自行安装，仓库不附带。

下面的命令使用 rootless Podman（`--group-add keep-groups` 等参数是 Podman 特有的）；用 Docker 的用户需要自行调整设备挂载和用户映射方式，仓库未在 Docker 上验证过启动流程。

**1. 构建自定义算子**（容器内挂载仓库到 `/project`，可写构建目录到 `/work`，禁用镜像不兼容的 Triton）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORK_DIR=/work/qwen36-build bash /project/scripts/build-native.sh
```

构建固定社区 commit，应用本仓库算子快照，产出独立的 `qwen36_int4` Torch namespace，不覆盖官方 `_C_ascend`。分步构建已验证，但公开脚本尚未在全新环境端到端逐字验收过。

**2. 下载并校验权重**

```bash
ms-hub download ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P --local-dir ./weights
python3 tools/validate_checkpoint.py ./weights --verify-hashes
```

**3. 启动服务**（宿主用 rootless Podman；下面是 TP1/MTP1/4K 的首次复现配置）

```bash
MODEL_DIR=/absolute/path/weights \
OPP_DIR=/absolute/path/qwen36-build/artifacts/opp/vendors/custom_transformer \
EXTENSION=/absolute/path/qwen36-build/artifacts/qwen36_int4_custom.so \
TP=1 MTP=1 MAX_MODEL_LEN=4096 MEMORY_UTILIZATION=0.90 PORT=18015 \
bash scripts/run-service.sh
```

`MTP=0` 关闭推测解码，`TP=2` 用两颗芯片。长上下文需要单独做内存和质量验证，不要直接把 4K 配置套用到 112K。

**4. 验证生成**

```bash
curl http://127.0.0.1:18015/health
python3 tools/bench_generation_sanity.py --base-url http://127.0.0.1:18015 --model qwen36-w4a8 --out /tmp/generation.json --max-tokens 128
python3 tools/bench_http_stream.py --base-url http://127.0.0.1:18015 --model qwen36-w4a8 --prompt-tokens 128 --output-tokens 256 --concurrency 1 --repeats 3 --out /tmp/stream.jsonl
```

sanity 脚本只检查服务可用、输出非空、无明显异常，不是准确率评测。benchmark 计时含 HTTP/调度开销。

## 生产档配置（long-prefill）

上面是最小复现路径；长上下文场景用 `scripts/run-candidate.sh`，它额外接入了 MindIEQ Prefill 分流和 C8。三档可选，`long-prefill` 是唯一对外推荐档，另外两档仅用于内部诊断对照：

| profile | MTP 草稿数 | 用途 |
| --- | ---: | --- |
| `compat` | 1 | 兼容性基线、问题定位 |
| `decode` | 2 | 内部 Decode 候选对照 |
| `long-prefill` | 2 | 对外长上下文档，启用 MindIEQ，短输入按阈值回退原 W4 |

```bash
MODEL_DIR=... OPP_DIR=... EXTENSION=... C8_LIBRARY=... GDN_OPP_DIR=... GDN_LIBRARY=... GDN_SOURCE=... MINDIEQ_LIBRARY=... \
bash scripts/run-candidate.sh --check   # 只校验制品哈希和参数，不初始化 NPU
bash scripts/run-candidate.sh           # 真正启动，默认 long-prefill
```

`QWEN36_C8_ENABLED=1` 可显式开启 C8；详细参数矩阵、各 profile 的选择依据和长上下文容量核算见[Profile 说明](profiles-20261014.md)和[部署矩阵](deployment-matrix-20261014.md)。

MindIEQ 长 Prefill 分流相对原始 W4A8（同输入交替五次中位数）：8K Prefill +23.29%、32K +20.69%，Decode 变化小于 0.5%。详细数据见[MindIEQ 评估记录](mindieq-evaluation-20261014.md)。

MTP2 相对 MTP1（112K Graph，32K 输入/256 输出，同协议）：Decode 23.20 → 28.94 tok/s（约 +24.8%），Prefill 约 −0.8%。并发 2/4 下总输出吞吐都约 5.06 tok/s，说明并发瓶颈在 Prefill/调度而不是 Decode。详见[Prefill/Decode 记录](prefill.md)。

## 量化与计算路径

仅 routed experts 量化为 signed INT4（ModelSlim 对称 per-channel SSZ）；scale 按 G128 kernel 布局重复存储，不是独立 group-wise 量化。激活是设备端 per-token INT8，运行时用两组 INT4 limbs 组合做 Cube 整数计算。非 expert 张量和 MTP 的 19 个张量保留原始浮点。目前只验证了纯文本生成。来源与许可见[provenance](../provenance.md)和[NOTICE](../../NOTICE)。

重新量化：`tools/convert_modelslim_w4a8.py` 复用已安装的 ModelSlim SSZ 产出 expert sidecar，不覆盖原始权重；`tools/export_checkpoint.py` 把 sidecar 和源模型合并导出成自包含 checkpoint，逐 tensor 核对 SHA256。导出后用 `validate_checkpoint.py --verify-hashes` 复查，再跑一次 MTP 和生成验证。

## C8（INT8 KV）

独立实验后端，走 `experimental/run-c8-service.sh`，需要额外构建 C8 库，不能直接给标准启动器加 C8 参数。已验证：TP1、eager/Graph、MTP1/MTP2，4K/32K/112K 全部通过启动、数值和长上下文 needle 检索。未验证：TP2、正式 P95、32G 实卡下的并发批处理。

细节见[C8 独立复现记录](c8.md)（选页 NZ 转换、原生 PA 对照、Cube/Split-KV 候选）和[Graph 上游交接记录](graph-upstream-patch.md)（310P Graph 生命周期差异、提交上游前的代码边界）。[参数选择记录](parameter-selection-20261012.md)记录了 C8/C16 和 MTP0/1/2 的取舍依据。

## 延伸阅读

- [整体思路历程](development-history.md) — 适配过程的技术叙事
- [代码入口与维护边界](../code-map.md) — 各目录职责和维护原则
- [发布校验记录](release-validation.md) — ModelScope 上传和独立加载验证的完整记录
- [GDN 状态行 bug：复现与上游修复](gdn-bug-fix.md) — 最小复现、补丁、上游 PR #18730 的完整过程
- [MindIEQ 长 Prefill 量化方案](mindieq-quantization.md) — 从微算子验证到输入规模分流的技术演进
- [业务负载回放：W8A8 TP2 与 W4A8 双实例](business-replay-20261031.md) — 分阶段真实调用形状下的对比，结论是维持 W8A8 TP2

原始实验数据和过程记录（多机调参历史、各轮复测）保存在 `results/qwen36/`，按日期归档，不改写；需要追溯具体某次实验时查对应目录。
