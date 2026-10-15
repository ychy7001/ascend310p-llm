# RC MTP2 第二draft slot索引：CPU精确算术隔离

基于`qwen36-mtp2-snapshot-draft-eager-20261015`，新增独立候选
`qwen36-mtp2-slot-cpu-20261015`。保留accepted-token快照补丁；target Graph、
draft eager、C16、同步调度、GDN/MindIEQ、max32768、KV3GiB、batch2048不变。
没有沿用上一轮runner分段同步屏障。

只挂载替换`llm_base_proposer.py`：在`attn_update_stack_num_spec_norm`中把
第二个及以后draft的slot位置向量复制到CPU，执行原来的整数`//`与`%`，
再阻塞上传商和余数；block table gather、slot写回和模型计算保留NPU。
保留原dtype和有符号整数语义，不引入浮点近似。

此为隔离候选，有额外D2H/H2D和同步；明确拒绝draft Graph，不是可直接发布的
通用修复。不修改默认服务、镜像或驱动。首次执行打印`RC_SLOT_CPU`，记录
block size、dtype、shape和位置值。完整差异见`proposer-slot-cpu.patch`。

验收先短题8/8及A/B/A，再32K warmup+2次冷请求、长请求后质量检查。
若失败则不做吞吐测试。结束恢复原MTP1并复验。

首次实际输入：block_size=128、dtype=int32、shape=[1]、positions=[27]，
见`slot-input.txt`。短题8/8、A/B/A一致、无canary泄漏，见`quality-before.json`。
保持同一target Graph/draft eager组合，仅替换slot商余数后短门禁通过，
支持继续定位该路径，但CPU复制引入同步，尚不能完全排除时序变化的影响。

## 32K冷请求与后置检查

精确32256-token输入，warmup一次（32-token输出）、两次计时（各128-token输出），
独立token前缀和cache salt。请求阶段计数，不是纯kernel耗时。

| 计时轮次 | Prefill tok/s | Decode tok/s | draft接受率 | prefix hit |
|---|---:|---:|---:|---:|
| 1 | 505.56 | 20.94 | 78.0% | 0 |
| 2 | 507.99 | 21.89 | 83.3% | 0 |

原始结果`bench-p32256.jsonl`。日志明确记录`Replaying aclgraph`，
见`graph-status.txt`。32K后短题8/8、A/B/A一致，无canary泄漏。
这不是与MTP1匹配条件的性能A/B，也不是正式精度评测或112K验收。

32K会议决议更新canary四字段全部正确（预算180000、周启、2026-11-12、延期上线），
54-token输出，Prefill441.28 / Decode23.92 tok/s，draft接受36/36。
见`meeting-32256.json`。该输入与合成重复文本不同，不直接比较吞吐。

工程结论：未升级驱动即可通过本轮MTP2 target Graph/draft eager的32K检查，
局部索引兼容方案有效，值得进一步减少CPU往返。不能据此证明驱动无关，也不能
排除CPU复制的同步时序影响。下一步优先验证block128、有界位置域的设备端精确
替代，再尝试draft Graph和112K；默认MTP1保持不变。

原MTP1服务已恢复：health200、短题8/8、A/B/A一致、无canary泄漏，
见`restored-mtp1-quality.json`。仅`qwen36-orangepi`运行，候选停止保留。
