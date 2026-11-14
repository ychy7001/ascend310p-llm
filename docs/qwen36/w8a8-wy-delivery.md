# Qwen3.6 W8A8：双 310P 默认交付

本仓库维护内部推理引擎的交付源码、registry 版本、默认部署配置与验收记录。双 310P 默认使用 **Qwen3.6-35B-A3B-W8A8 + native WY + MTP2 + Decode Graph + eager mask 复用**，适用于所有重 Prefill 业务。20261114 版本已通过本次性能、精度与封装门禁，发布到 registry，并替换仓库默认交付。

## 拉取与配置

```bash
podman pull registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:75a20b2954edae93e5990f795e12c09c06faadcc49d6acd880cbc2f5f8ab217d
```

| 项目 | 本次交付 |
| --- | --- |
| 固定版本 tag | `qwen36-tp2-w8a8-wy-mtp2-20261114` |
| 默认版本发现 tag | `qwen36-tp2-w8a8-wy-mtp2-default` |
| Registry | `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend` |
| 固定 digest | `sha256:75a20b2954edae93e5990f795e12c09c06faadcc49d6acd880cbc2f5f8ab217d` |
| Config / image ID | `2793814b8ad397766b20518e09e60c1a3066f5ebacb0cb71f02601916bd018f6` |
| 配置 | TP2、262144 context、2048 batch token、并发上限 4、MTP2、target Graph `[3,6,12]`、draft eager、prefix cache 与 async 开启 |
| 已测硬件与驱动 | a2 双 310P3，驱动 26.1.1；镜像 CANN 9.1.0-beta.1 |
| 正式复用源码 | `runtime/wy_compat/eager_mask_reuse.py` |
| 机器可读发布记录 | [release.json](../../results/prefill-mask-reuse-20261114/release.json) |

部署锁定 digest；`default` tag 用于发现当前发布版本。旧的日期版本保留供回退。启动入口、环境变量、Quadlet 和 manifest 都随默认登记同步维护，参照[部署步骤](w8a8-262k.md)。

## 配套模型

镜像交付运行时；配套权重为通过验收的 ModelSlim W8A8 checkpoint，served name `Qwen3.6-35B-A3B-w8a8`。已逐文件校验的 a2 内部模型目录：

```text
/srv/meetai/models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8
```

该目录 22 个文件与本次验收副本 SHA256 完全一致，其中含 10 个权重分片。完整权重、tokenizer 和配置身份见 [w8a8-checkpoint.sha256](../../results/prefill-mask-reuse-20261114/w8a8-checkpoint.sha256)。在目标权重目录运行：

```bash
sha256sum -c /path/to/ascend310p-llm/results/prefill-mask-reuse-20261114/w8a8-checkpoint.sha256
```

全部匹配后把该目录只读挂到 `/model`；从仓库环境模板配置宿主路径。权重单独分发，不烘焙进引擎镜像。

## 放行依据

- 性能：同一 WY 基线、同模型与 CLI 的三轮 32K 冷输入对照，首 token 中位数 **22.521 → 20.942 秒（−7.01%）**，服务端 Prefill 吞吐 **+7.55%**；4K 基本持平。见[性能记录](../learning/cold-prefill-06.md)。
- 精度：双 rank 合计 **540 次实际 SplitFuse 完整 FP16 输出逐位相同**，其中 360 次多请求，覆盖 29 组有序形状、最大 context 32774。原路径自身的跨轮差异已通过 fresh/fresh 对照复现；见[精度门禁](../learning/cold-prefill-07.md)。
- 封装：同一已测源码 SHA；入口和实际 CLI 继承；32K 检索正确、双 rank 命中、Decode Graph 捕获与重放、容器退出 0、a2 清场。原冒烟脚本因 Graph 日志文字误配而退出 1，保留原状态后按实际 WY 日志重新判定并修正脚本；见[封装记录](../../results/prefill-mask-reuse-20261114/packaged-image.json)。
- 发布：固定版本 tag、default tag 的远端 manifest 一致，config digest 对应验收 image ID；固定 digest 和 default tag 均已实际 pull 校验。见[发布记录](../../results/prefill-mask-reuse-20261114/release.json)。

本次放行针对本机型、固定配置和上述门禁。262K 容量/PDF 等能力的历史验收来自原 WY 基础镜像；本次增量数值门禁覆盖至 32K，没有重复整套容量测试。新增极长上下文、不同权重或不同量化方案时，以新交付版本补充相应门禁。

## 回退

优先在同档位回退，以保留模型与启动配置：

```text
registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5
```

只需关闭复用时，在环境文件设置 `QWEN36_EAGER_MASK_REUSE=0` 并重启；镜像默认值为 `1`。整镜像回退则将 Image/manifest 的 image 一起改回上述 digest。切换按现有设备停服、空闲检查及 60 秒冷却流程执行。

仓库默认交付已更新；各业务机器何时重启加载新版本由站点部署流程执行，本次发布未改动 a1 在线服务。

## 后续发布规则

1. 候选验证阶段记录性能、精度与封装结果。通过后建立固定版本 tag，核验远端 manifest/config 并按 digest 拉回。
2. 发布者更新模型专用 `default` tag、`deploy/variants/qwen36.json`、所有当前部署模板与本页，保留前一版 digest 及验收依据。
3. 执行 `python3 tools/check_default_delivery.py`，提交与推送交付源码和发布记录。已通过且发布的版本标为 `default-delivery`；后续研发另行保留实验状态。
4. 发生回归时按同档位 digest 回退，并同步恢复默认登记与 default tag；保留失败版本的诊断记录。
