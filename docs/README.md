# ascend310p-llm 文档索引

仓库按"主线 + 方法 + 制品 + 研究 + 回馈"分层。`results/` 放原始采集，`docs/` 只放结论与协议。

## 交付主线

| 线 | 入口 | 状态 |
|---|---|---|
| Qwen3.6-35B-A3B（双 310P，W8A8/TP2） | [qwen36/README.md](qwen36/README.md) | 生产交付 |
| Qwen3.8-Flash-Next（单张 300I DUO，TP2/EP2） | [qwen38/README.md](qwen38/README.md) | 试验 |
| ASR（Confucius4-R2T2 / Qwen3-ASR，单颗 310P3） | [asr-310p-versioning.md](asr-310p-versioning.md)、[confucius4-r2t2/bring-up.md](confucius4-r2t2/bring-up.md) | 实验，未接入生产 |

Qwen3.6 的具体文档（交付、复现、Graph、MTP、质量、部署矩阵）都在 `qwen36/`，以该目录 README 为准。生产默认档：[qwen36/w8a8-wy-delivery.md](qwen36/w8a8-wy-delivery.md)；满上下文配置：[qwen36/wy-mtp-production-20261108.md](qwen36/wy-mtp-production-20261108.md)。

## 通用方法与测量

- [ascend310p-baseline.md](ascend310p-baseline.md) — 平台身份与测量协议（换机、换版本前先看）。
- [ascend310p-optimization-plan.md](ascend310p-optimization-plan.md) — 优化优先级与"什么算采用"的条件。
- [ascend310p-profiling.md](ascend310p-profiling.md) — 采集方法与算子证据口径。
- [ascend310p-results.md](ascend310p-results.md) — 310P 优化结果汇总。
- [code-map.md](code-map.md) — 代码入口与维护边界。
- [project-summary.md](project-summary.md) — 项目技术总结。

## 制品、来源与精度契约

- [production-image-source.md](production-image-source.md) — 固定来源的生产镜像构建。
- [provenance.md](provenance.md) — 来源、修改与精度契约。
- [production-image-base-evaluation.md](production-image-base-evaluation.md) — 官方基础镜像替换评估（2026-11-05，只读记录）。

## 研究

[research/](research/) 放跨主线的取舍与定位：CANN 升级评估、GDN 上游候选、SplitFuse 因果边界、WY × compressed attention、模型业务质量对照、镜像基准判定。

## 上游回馈

[upstream/README.md](upstream/README.md) — 310P legacy SplitFuse 修补的上游提交与对照协议。

## 学习记录

[learning/cold-prefill-01.md](learning/cold-prefill-01.md) 起共 7 篇，记录从调度预算到 WY 算子、再到 mask 复用的推理过程。属于背景材料，不作为交付依据。

## 约定

- 结论写进 `docs/`，原始 profile / 日志进 `results/<实验名>/`，两者用报告名对齐。
- 版本或底座变化时，先更新 `production-image-source.md` 与 `provenance.md`，再改主线文档。
- 证据标签：source-code verified / documentation verified / issue-PR-commit / experimentally verified / hypothesis-inference。
