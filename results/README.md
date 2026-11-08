# results：原始数据索引

按实验线分组，组内按时间排序。目录名带日期，不改写。每个目录的 `README.md` 是该次实验的结论与限制；本页只做定位。

`results/qwen36/` 和 `results/qwen38/` 是多轮子运行的历史归档，条目多，单独列在文末。

**其中 17 个目录只在实验机上留存、未纳入 Git**（`.gitignore` 按整目录忽略，多为大体积日志、CSV、trace 和权重清点）。下表标了「本机留存」；clone 后在仓库里看不到这些目录，需要原始文件时回保留它们的机器查。相关文档已按此说明标注。

## W8A8 / native WY 生产化（2026-11-03 起）

| 目录 | 内容 | 对应文档 |
| --- | --- | --- |
| `production-w8a8-20261103/` | 生产 W8A8 TP2 副本基线 | [W8A8 复现](../docs/qwen36/w8a8-262k.md) |
| `production-native-wy-20261103/` | 原生 WY 单变量实验，持续更新 | [W8 生产配置](../docs/qwen36/wy-mtp-production-20261108.md) |
| `production-wy-control-recheck-20261103/` | WY 配对复核 | 同上 |
| `production-wy-batching-recheck-20261103/` | WY 并发、数值拆分与候选 profile | 同上 |
| `wy921-gate-20261104/` | WY 9.2.1 模型级门禁 | [CANN 升级评估](../docs/research/cann-upgrade-evaluation.md) |
| `wy921-localize-20261104/` | WY921 第三轮：真实运行时偏差定位 | 同上 |
| `wy921-fix-20261105/` | WY 原生实现修复与完整门禁（wy-4） | [WY 完成报告](../docs/qwen36/wy-optimization-completion-report-20261108.md) |
| `wy921-reference-20261105/` | wy-5 高精度参考与 logprob 比较器可行性 | 同上 |
| `wy921-boundary-20261106/` | WY 失败点的边界诊断 | 同上 |
| `wy921-guarded-rounding-20261106/` | WY 保护式舍入诊断 GRC1 | 同上 |
| `native-wy921-20261104/` | 9.2.1 原生 WY 重编译与两设备门禁 | [CANN 升级评估](../docs/research/cann-upgrade-evaluation.md) |
| `cann-wy-feasibility-20261104/` | CANN compute-WY 可行性核查 | 同上 |
| `upstream-gdn-pr-audit-20261104/` | 上游 GDN/WY 差异审计 | [GDN bug 修复](../docs/qwen36/gdn-bug-fix.md) |

## W8A8 / MTP 交付（2026-11-08）

| 目录 | 内容 | 对应文档 |
| --- | --- | --- |
| `wy-mtp-production-20261108/` | W8 WY/MTP2 满上下文配置资产 | [生产配置](../docs/qwen36/wy-mtp-production-20261108.md) |
| `wy-mtp-integration-20261108/` | WY/MTP 组合验证的多轮子运行（无顶层 README，按子目录查） | [组合验证](../docs/qwen36/wy-mtp-integration-20261108.md) |
| `wy-mtp-mask-fix-20261108/` | SplitFuse mask Graph 兼容修复的多轮子运行（无顶层 README，按子目录查） | [mask 修复](../docs/qwen36/wy-mtp-mask-fix-20261108.md) |
| `wy-registry-delivery-20261108/` | registry 推送登记、远端 manifest 原始字节、拉取核验 | [交付记录](../docs/qwen36/wy-optimization-completion-report-20261108.md) |

## WY 数值诊断（2026-11-07）

| 目录 | 内容 |
| --- | --- |
| `wy-fp64-diagnosis-20261107/` | WY FP64 分阶段诊断 |
| `wy-gram-postbeta128-20261107/` | raw-K Gram 后乘 beta 候选 |
| `wy-postbeta-neumaier-20261107/` | raw-K Gram + Neumaier 设备扫描 |
| `wy-postbeta-refined-20261107/` | raw-K Gram + Neumaier + 全 128 维 refined 求解 |
| `wy-postbeta-refined-screen-20261107/` | 全 refined WY 固定五点模型筛查：不通过 |
| `wy-reducesum-runtime-audit-20261107/` | WY ReduceSum runtime 审计 |
| `wy-torch-ut-20261107/` | WY Torch UT 隔离候选 |
| `wy-torch-ut-graph-20261107/` | 原 Torch UT 算术的 NPU 图回放探针 |
| `wy-torch-wy-fullgraph-20261107/` | 完整原 Torch WY 图回放：较长形状无收益 |
| `operator-feedback-20261107/` | 外部算子调查反馈的逐节复核（`assessment.md`） |

## GDN 配套 H/O 与整栈对照

| 目录 | 内容 |
| --- | --- |
| `upstream-gdn-pair-v2-20261104/` | 固定 PR17640 成对 FwdH/FwdO 门禁 |
| `upstream-gdn-pair921-v1-20261104/` | CANN9.2.1 配套 H/O 独立正确性验证 |
| `upstream-gdn-cann921-pair-repro-20261104/` | 310P / CANN9.2.1 配套 H/O 对照复现（未公开发布） |
| `upstream-gdn-pair-v1-20261104/` | 成对门禁首轮，仅构建日志与状态（本机留存，未入库） |
| `gdn-runtime-control-20261104/` | GDN 跨运行栈对照 |
| `production-gdn-pair-quality-20261104/` | 配套 GDN H/O + 原生 WY 模型筛选 |
| `action-state-paired-v1-20261104/` | 结构化字段与长上下文配对验证，结论见 `adjudication.json` |
| `short4-recheck-20261104/` | short-4 有限复核 |
| `a2-recovery-20261104/` | 原服务恢复记录 |

## CANN 9.2.1 升级评估

| 目录 | 内容 | 对应文档 |
| --- | --- | --- |
| `cann921-upgrade-20261104/` | CANN9.2.1 完整官方栈隔离验证 | [CANN 升级评估](../docs/research/cann-upgrade-evaluation.md) |
| `cann-upgrade-audit-20261104/` | 官方 release/安装文档原文与镜像 manifest 归档（本机留存，未入库） | 同上 |

## Decode 归因与 Graph 候选

| 目录 | 内容 |
| --- | --- |
| `decode-attribution-20261104/` | Decode 归因 v3：8/8 设备窗口、768 步稳定区间；历史分析在 `v1-README.md` |
| `decode-candidate-audit-20261104/` | Decode 候选只读审计 |
| `production-graph14-20261103/` | Graph `[1,4]` 候选定义与冻结（本机留存，未入库） |
| `production-graph14-retry-20261103/` | 同候选重试与业务事实采样（本机留存，未入库） |

## Dense 动态量化

| 目录 | 内容 |
| --- | --- |
| `dense-quant-isolation-20261103/` | Dense 动态量化矩阵乘合成隔离 |
| `dense-quant-captured-replay-20261103/` | 第一次真实输入重放：准备失败 |
| `dense-quant-captured-replay-v2-20261103/` | 真实 down_proj 输入独立重放 |
| `dense-quant-allocator-paired-20261104/` | 实际 rank1/device0 输入，分配器交替重复 |
| `dense-quant-synthetic-allocator-paired-20261104/` | 不含模型权重的随机 INT8/零输入 507015 复现 |
| `dense-quant-expandable-grid-20261104/` | 真实 Matmul 输入，显式 expandable 分配器交叉 |
| `dense-quant-rank-device-grid-20261104/` | 两个 TP 分片 × 两张物理 NPU 独立重放 |
| `dense-runtime-audit-20261104/` | Dense 后续兼容性镜像审计 |

## Qwen3.8-27B Dense

| 目录 | 内容 |
| --- | --- |
| `qwen38-27b-w8a8-310p-20261103/` | 权重清点与下载、配置预检（本机留存，未入库） |
| `qwen38-dense-baseline-20261103/` | Dense 首次启动记录 |
| `qwen38-dense-baseline-quant-trace-20261103/` | 量化路径 trace 采集（本机留存，未入库） |
| `qwen38-dense-baseline-sync-diagnostic-20261103/` | 同步点诊断（本机留存，未入库） |
| `qwen38-dense-baseline-sync-eager-diagnostic-20261103/` | 同步点诊断 eager 对照（本机留存，未入库） |
| `qwen38-dense-metadata-20261103/` | 权重元数据审计 |
| `qwen38-dense-no-executor-cache-screen-20261103/` | `ACLNN_CACHE_LIMIT=0` 启动 screen |
| `qwen38-dense-post4-runtime-v1-20261104/` | post4 runtime screen v1 |
| `qwen38-dense-post4-runtime-v2-20261104/` | post4 runtime screen v2 |
| `qwen38-dense-post4-runtime-v3-20261104/` | post4 runtime screen v3 |
| `qwen38-dense-post4-runtime-kv94-20261104/` | post4 完整上下文容量与短正确性筛选 |
| `qwen38-dense-quant-repack-20261104/` | 模型内 down_proj 权重存储重建诊断 |

## FwdO 候选与 Profile 采集

| 目录 | 内容 |
| --- | --- |
| `fwd-o-rebuild-control-20261103/` | 原始 FwdO 独立重编译对照 |
| `fwd-o-maskfix-fences-rejected-20261103/` | FwdO fence 补丁失败记录 |
| `fwd-o-maskfix-vector-rejected-20261103/` | FwdO vector 补丁失败记录（本机留存，未入库） |
| `fwd-o-isolation-20261103/` | FwdO 隔离重编译状态（本机留存，未入库） |
| `upstream-fwd-o-20261103/` | 上游 FwdO 单独替换组合：未通过 |
| `executor-cache-control-20261103/` | `ACLNN_CACHE_LIMIT` 原始 GDN 对照 |
| `prefix-c16-20261103/` | C16 prefix 复用尝试的启动日志（本机留存，未入库） |
| `profile-c16-20261103/` | C16 kernel CSV、HTTP 采集与前后 quality（本机留存，未入库） |

## 审计与证据

| 目录 | 内容 |
| --- | --- |
| `audit-20261103/` | 只读环境与镜像 manifest 原始证据 |
| `production-a1-audit-20261103/` | 生产基线只读审计与隔离复现 |
| `production-kernel-source-audit-20261103/` | kernel 源码与适配层逐文件对照（本机留存，未入库） |
| `deployment-profile-audit-20261104/` | a2 设备快照（本机留存，未入库） |
| `evidence-protocol-20261104/` | 证据行动项协议最小原型 |
| `evidence-protocol-20261104-prepare/` | 冻结 fixture 准备（本机留存，未入库） |
| `evidence-protocol-20261104-prepare-v2/` | 冻结 fixture + 校验器变异体（本机留存，未入库） |
| `baseline-20261103/` | 单流 baseline 原始 jsonl，1K–32K 四档（本机留存，未入库） |

## 其他

| 目录 | 内容 |
| --- | --- |
| `pdf-ocr-baseline-20261107/` | PDF OCR Torch 基线冒烟 | 
| `local-gpu-reference-20261104/` | 本机 GPU 现状与 27B 现有服务对照 |
| `local-gpu-business-20261104/` | GPU 27B 业务与字段诊断 |

`results/` 根目录另有若干 `upstream-*-repro-*.tar.gz` 与同名 `.manifest.json`：对外复现用的打包快照，manifest 记文件哈希。

## 历史归档

| 目录 | 内容 |
| --- | --- |
| `qwen36/` | Qwen3.6 全部历史子运行，72 个日期目录：C8、MindIEQ、MTP2 定位、RC、orangepi 部署等。无顶层 README，各子目录自带 |
| `qwen38/` | Qwen3.8 Flash-Next 的 7 个子运行：TP2 可行性、serving、decode graph、长上下文、32K 边界、prefill batch512 |

`qwen36/` 里的早期子目录（`mindieq-r3-20261013`、`persistent-nz-20261015`、`pr17640-b4096-20261013`、`pr17640-product-20261013`）没有 README，按目录里的 `status.json` / 日志文件名判读，不再单独补索引。
