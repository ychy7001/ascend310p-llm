# W8 WY / MTP Graph mask兼容修复（2026-11-08）

目标是在W8当前CANN 9.1.0-beta.1、torch_npu 2.10.0.post2及WY镜像9df3c6f3…上修复MTP Graph，不替换WY数学路径，不触碰a1或已发布镜像。

## 根因与修复范围

两份镜像的310P attention_v1.py相同，但W4为torch_npu post4 / CANN 9.1.0，支持压缩SplitFuse v2；W8为post2 / beta.1，缺少对应Python接口，使用旧SplitFuse路径，mask生成会将query_start_loc和seq_lens复制到CPU。在捕获中该拷贝失败。依赖与源码哈希见`results/wy-mtp-mask-fix-20261108/dependency-comparison.json`。

直接在Graph内生成mask仍会触发旧栈整数Sub的aclop捕获限制。因此候选在ACLGraphWrapper进入捕获或replay之前，用当前设备长度生成完整因果mask并写入地址固定的NZ缓冲区；捕获内部只返回该缓冲区。每次replay都刷新内容，不能只缓存首次capture的mask。

mask列宽固定为模型max_seqlen；原有上下文范围内的0/-inf可见性相同，额外未来列均为-inf。query布局定义Graph缓存描述，设备seq_lens决定当前可见范围。普通eager和Prefill路径仍调用原mask函数。该适配器改变运行时mask准备位置和缓冲区管理，不改变WY kernel、阈值、协议、规则、worker或R。

## 已结束的小规模验证

- mask-replay-v1：在Graph内生成mask失败，错误指向Sub的aclop加载，已独立清场。
- mask-replay-v2：在Graph外更新固定缓冲区，Graph读取该缓冲区；query布局[1]、[3]、[3,3]、[1,3]，分别改变五组上下文长度，20/20满足可见范围与未来屏蔽不变量。已独立清场。
- 该小测验证固定缓冲区的更新与读取，不等于完整attention或整模型已验收。
- 首轮整模型使用TP2、mode0/FULL_DECODE_ONLY、MTP2主模型Graph/草稿eager、seq1、batch2048、max114688。启动后才进入已知答案、4K/32K冷输入、128输出与PDF smoke；以实际结束结果为准。

脚本及候选代码位于`experimental/wy-mtp-20261108/splitfuse-fix/`与`run_w8_mask_fix_20261108.py`，尚未写入发布镜像。

## 整模型诊断结果

`w8-mask-fix-20261108-v1`完成，总耗时408.011秒，driver 0、容器正常退出，独立清场通过。三条已知答案正确；实际Graph捕获后正常进入请求。

| 测试 | 结果 |
| --- | --- |
| 4K冷Prefill，3次 | 3/3有效，中位2015.305 tok/s，prefix hits均0 |
| 32K冷Prefill，3次 | 3/3有效，中位1418.069 tok/s，prefix hits均0 |
| 4K/128 Decode，3次 | 3/3有效，中位47.006 tok/s |
| MTP2 | 每次94草稿token接受80，接受率85.106% |
| PDF四页各一次 | 功能检查4/4，pdf-01达到512输出上限，其余正常stop |

性能为该配置的绝对实测，没有同窗口无MTP对照，不计算提升百分比。PDF功能检查不是完整OCR质量验收；多模态缓存未专门禁用，不称为完全冷视觉吞吐。

归档`results/wy-mtp-mask-fix-20261108/model-diagnostic/`含summary、压缩原始测量、manifest、image/source freeze、smoke、PDF、日志与清场。候选源码与bootstrap的适配器SHA256均为47e1dee2f55d46a358eabe3d9487364a62e7e3d4c65fd54d0b2a9e665429592f。

已封装独立镜像，正在验证默认入口；在该验证结束前不以挂载源码的诊断通过替代镜像交付验收。当前修复只声明主模型Graph + MTP2草稿eager；未声明多步草稿Graph兼容性。
