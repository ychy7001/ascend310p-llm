# RC MTP2：草稿 Graph 与请求历史对照

目的：区分第二草稿接受率偏低是否依赖草稿 Graph，或在长 Prefill / 请求复用后才出现。
本轮不改稳定运行时，不升级驱动，不将诊断补丁合并为默认。

## 固定条件与协议

- 主机：orangepi，ARM64 / RC / 310P1，共享主机内存；不是物理48GiB RC验证。
- TP1、W4A8、C16、MTP2、target Graph、GDN、MindIEQ、同步调度、prefix/视觉开启。
- max_model_len=114688，batch2048，expert chunk512，KV逻辑预算4GiB。
- 沿用accepted-token快照与block128设备slot算术补丁。
- 仅切换 `MTP_DRAFT_EAGER=0/1`；容器镜像、挂载、命令和其余环境一致。
- 顺序运行，每档独立启动：8题+A/B/A、3K/32预热、两份冷32K/128请求、
  32K会议canary、重复同两份冷32K/128请求、8题+A/B/A。
- 两档复用同一token-ID文件。每个测量使用独立cache salt，核对prefix hit增量0。
- 每次记录两个位置接受数及draft调用数。逐位置接受率分母为draft调用数；
  不是两位置合并的接受率，也不表示所有第二位置均有条件接受机会。

只预热3K，未先执行完整32K预热；两档协议相同，但首轮32K应单独看待。
每阶段仅两份输入、每份一次；不是P50/P95、完整业务精度集或稳定性能承诺。
会议canary是四字段功能检查，不能代替长输出Decode测量。

## 运行与数据

`run.py` 是实际主机编排脚本，依赖上一轮目录
`/data/qwen36-deploy/results/mtp2-fullgraph-long-20261015/` 内冻结的工具和输入。
它只创建独立候选、保留已有容器；完成或异常后恢复 `qwen36-orangepi` 并复验。
此脚本使用本轮固定容器名/输出目录，重跑须更换目录和名称，不能直接覆盖。

```bash
# 主机运行入口（本轮已执行，不重复启动）
python3 -u /data/qwen36-deploy/results/mtp2-history-20261016/run.py
# 仓库内验证并汇总实际完成的数据
python3 results/qwen36/mtp2-history-20261016/summarize.py
```

`state.txt`、`run.log` 是归档时进度快照，不代表后续实时状态。
尚未完成的结果不填入报告。最终恢复以 `restored-mtp1-quality.json` 为准。

## 与旧32K档的区别

`prior-config-diff.json` 保存实际Docker配置核对：镜像ID和命令相同，
挂载集合相同（列表顺序不同）；环境差异为max_model_len从32768到114688、
KV逻辑预算从3GiB到4GiB。请求顺序也不同。
因此旧32K档约23tok/s不能直接作为本轮112K档的匹配基线，
也不能据此直接归因于Graph、驱动或长Prefill状态损坏。

## 结果

| 阶段/输入 | 全Graph Prefill / Decode tok/s | 全Graph第二位置接受率 | draft eager Prefill / Decode tok/s | draft eager第二位置接受率 |
|---|---:|---:|---:|---:|
| 会议前/1 | 508.09 / 17.06 | 2/66 = 3.03% | 506.87 / 15.49 | 0/68 = 0% |
| 会议前/2 | 508.78 / 17.01 | 1/66 = 1.52% | 508.28 / 16.27 | 3/65 = 4.62% |
| 会议后/1 | 508.20 / 16.58 | 1/68 = 1.47% | 506.44 / 21.16 | 37/50 = 74.00% |
| 会议后/2 | 509.88 / 16.74 | 0/67 = 0% | 508.16 / 16.00 | 1/66 = 1.52% |

8条请求均为32256输入、128输出、prefix hit增量0。两模式对应输入哈希和
完整输出四组全部一致，第二位置接受数之和与总接受数核对通过。
在同模式会议前后，输入1输出有一个措辞差异（phrase/sentence），输入2逐字一致；
不能声称所有请求历史下输出逐字确定。见 `summary.json` 和完整响应JSON。

两档会议canary均四字段正确：全Graph Prefill443.51 / 短答案Decode17.16，
draft eager 442.36 / 16.17tok/s。前后短题均8/8、A/B/A一致且无canary泄漏。

结论：

1. 第二位置在会议请求前已偏低，不支持将它简单描述为“长Prefill之后才损坏”。
2. draft eager同样出现低接受率，因此问题不只存在于草稿Graph。
3. eager会议后输入1单次恢复74%接受率，而下一条同阶段输入2仍低；
   存在值得复现的请求相关差异，但不能把单次恢复当作修复或平均性能收益。
4. Prefill两模式相近；不能靠关闭草稿Graph稳定解决MTP2 Decode问题。
   RC仍保留原C16/MTP1/Graph/GDN/MindIEQ、4096/512、112K默认。

下一步优先重复同一输入的冷A/B/A序列，核对第二草稿token/位置、KV block/slot、
GDN状态索引及接受后状态提交；记录关键张量值而非只记录Python对象存在。
将max_model_len和KV池大小分开测试仍有价值。上述只是排查优先级，尚未证明
具体状态污染、元数据别名、硬件故障或驱动缺陷，不同时混入驱动升级。

两候选已停止保留；原 `qwen36-orangepi` 已恢复，health200、8题8/8、
A/B/A一致且无canary泄漏，仅原稳定容器运行。见
`restored-mtp1-quality.json`；最终状态 `completed-restored`。
