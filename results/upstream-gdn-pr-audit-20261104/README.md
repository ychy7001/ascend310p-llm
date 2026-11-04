# 上游 GDN/WY 差异审计（2026-11-04）

仅只读查询 GitHub API，保存 PR 元数据、文件 diff 和 review；未发布评论/issue/PR。在线页面工具未返回可读正文，使用官方 GitHub API 原始快照。状态和 mergeability 随时间变化，以下为查询时值。

| PR | 状态 | head | mergeable / state |
|---|---|---|---|
| #17640 | open，未合并 | `18163f6b1d20` | True / unstable |
| #18057 | open，未合并 | `2620e77d7bdb` | True / unstable |
| #11941 | open，未合并 | `7b2bfbe0aecd` | False / dirty |
| #13122 | open，未合并 | `e82f62e3daad` | False / dirty |
| #17412 | open，未合并 | `6bf042277213` | True / unstable |
| #18410 | open，未合并 | `988fa61e81b9` | False / dirty |
| #8872 | open，未合并 | `e73276d07f06` | False / dirty |

用户提供的“至少六个310P相关Open PR”得到本次元数据支持。需要修正：本次 #17640 返回 mergeable=true / unstable，不能继续断言有合并冲突；#18057、#17412也返回true。unstable不等于已通过全部检查或可直接合入。

#18057 保存的人工 APPROVED review 对应旧提交111fd6c0，当前head2620e77d；存在人工批准历史，不等于当前head已得到重新批准。

## 与当前实验的关系

| 维度 | #17640 固定18163f6b | #18057 固定2620e77d | 本轮实验 |
|---|---|---|---|
| WY来源 | PR中的AscendC实现及独立binding | 外部CANN ops-transformer固定0340eae0，统一Python API | 直接复用#17640，未自研WY算法 |
| H/O | 成对优化，NZ状态交换，binding预置初始状态NZ图像 | 本次diff未修改H/O，主要WY/gating dispatch与包集成 | 已测WY-only保留生产H/O；现验证完整上游H/O对 |
| WY输入契约 | FP16 q/k/v/beta、FP32 g，chunk64，输出五张量 | 同类六输入/五输出调用；guard维度16倍数且<=128 | 保守限定D=64/128，T>0、整chunk、组头整除；TP2真实HK8/HV16/D128 |
| 加载与fallback | PR内native binding与Torch回退 | cann_ops_transformer.ops，缓存导入与函数查询，缺包/不支持形状回退Torch | 独立pr17640_wy namespace，固定生产模块SHA，CPU/不支持形状回退 |
| 额外改动 | H/O数值、mask、NZ等，另有默认关闭的Python优化 | 同时接入fused_gdn_gating并在调用前转FP16 | 本轮WY单变量未启用gating/host-commit/shared-quant |
| 环境证据 | 作者算子/整模型报告，多形状但非本轮生产栈 | Dockerfile当前基础CANN9.2.0，需外部run包与wheel匹配 | 生产CANN9.1-beta/post2；post4/CANN9.1另作完整镜像对照；未验证9.2兼容 |
| 性能可比性 | 局部BS4/TP4-rank形状、8K作者数据 | 不足以推导本业务TP2吞吐 | W8A8 TP2/C1零命中中位数，Prefill+38～54%；不与15.11x/8.32x直接比较 |

#18057不是完整FwdH/FwdO融合替换；其WY dispatch相似不能证明底层算子二进制等价。未取外部ops-transformer0340eae0全部源文件，因此尚未完成两者底层算法、支持范围和数值实现的逐行比较。该项不阻塞当前固定#17640成对H/O门禁。

本轮独立价值限于可复现的正确性诊断、生产形状回归以及后续能够证实的修复。GDN存在共享数值偏差；Dense507015也有无权重复现，但未证明属于vLLM-Ascend Python缺陷。两者都不能提前承诺BugFix PR。优先给现有PR补充验证/修复；只有独立根因和可审阅补丁才考虑另开PR。所有对外反馈仍未发布。
