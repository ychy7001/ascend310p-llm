# RC MTP2 draft eager：重复冷A/B请求

上一轮只有一条请求的第二草稿接受率恢复到74%，其余请求偏低。本轮使用
独立新进程复测，先不增加设备同步或运行时日志探针，检查该恢复是否可重复。

## 固定条件

orangepi / ARM64 / RC / 310P1；TP1、W4A8、C16、MTP2、target Decode Graph、
draft eager、GDN、MindIEQ、同步调度、prefix/视觉开；max114688、batch2048、
expert chunk512、KV逻辑预算4GiB，保留快照与block128设备slot算术补丁。
`config-check.json` 核对本候选与上一轮draft eager的环境、镜像ID、命令、
挂载集合全部一致。RC共享内存环境，不能当作物理48GiB RC整机验证。

## 协议

1. 独立启动，短题8/8和A/B/A复用检查。
2. 复用上一轮冻结token-ID，3K/32预热。
3. 第一组冷请求A/B，各32256输入、128输出。
4. 32K会议四字段canary，复用上一轮冻结输入。
5. 第二、第三组冷请求A/B，输入与第一组相同。
6. 短质量复验，停止候选，恢复原MTP1服务并复验。

六次测量使用独立cache salt，逐条核对prefix hit增量0，记录两个位置的
草稿接受数、draft调用数和完整输出。仅3K预热，没有完整32K预热；
分组顺序固定，单进程少量重复，不是随机交错试验、P95或正式质量评估。
额外第三组增加了请求历史，不能与上一轮第二组之后的不存在数据做对应性能比较。

## 文件与复现

`run.py` 为实际主机编排脚本，依赖上一轮冻结的
`/data/qwen36-deploy/results/mtp2-history-20261016/run.py` 及其依赖。
不属于部署入口。脚本使用固定容器名、输出目录，重跑需更换名称及目录，
不能覆盖已有结果。

```bash
# 主机入口（本轮已运行，勿重复启动）
python3 -u /data/qwen36-deploy/results/mtp2-repeat-20261016/run.py
# 仓库内核对实际完成结果
python3 results/qwen36/mtp2-repeat-20261016/summarize.py
```

`state.txt`、`run.log` 为归档时快照；恢复以
`restored-mtp1-quality.json` 和最终状态为准，未完成时不宣称恢复通过。

## 结果

六条冷请求全部完成，输入均32256 token、输出均128 token；六次prefix hit增量均为0。

| 顺序 | 输入 | Prefill tok/s | Decode tok/s | 第一/第二位置接受数 | draft次数 | 第二位置接受率 |
|---|---|---:|---:|---:|---:|---:|
| 第1组 | A | 506.85 | 15.55 | 60/0 | 68 | 0.00% |
| 第1组 | B | 509.38 | 15.73 | 60/0 | 67 | 0.00% |
| 第2组 | A | 506.89 | 15.29 | 59/0 | 69 | 0.00% |
| 第2组 | B | 509.34 | 15.67 | 60/0 | 67 | 0.00% |
| 第3组 | A | 507.28 | 16.95 | 55/11 | 62 | 17.74% |
| 第3组 | B | 509.34 | 15.74 | 60/0 | 67 | 0.00% |

第二组在会议canary之后，第三组紧接第二组。会议四字段全部正确，
Prefill442.33/短答案Decode16.06 tok/s；长请求前后短题8/8、A/B/A一致且无canary泄漏。

本轮没有复现上轮74%的第二位置接受率：五条为0，第三组A为17.74%。
Prefill稳定在506.85–509.38，Decode15.29–16.95 tok/s。
本轮不是新的MTP1配对对照，不将其与历史MTP1数据计算精确加速比。

同一输入B三次完整输出一致；A前两次一致，第三次把phrase改为sentence，
与上轮高接受那条输出一致，但第二位置接受率只有17.74%。因此仅凭最终输出
是否一致，无法解释draft接受率差异；需要逐步对齐draft/verify的token和元数据。
这些是观察结果，尚未证明状态污染、数值误差、Graph或驱动中的任何一项为根因。

## 决策与下一步

- RC继续采用已验证的C16/W4A8/MTP1/Graph默认档；不推广MTP2候选。
- 不再仅切换draft Graph或重复调batch来声称修复；下一轮用独立诊断候选，
  对齐每步两个draft token、target verify token、接受长度，以及positions、
  seq_lens、query_start_loc、slot/block映射；先找出第二位置首次偏离的步骤。
- 当前镜像`qwen3_5_mtp.py`第100–107行明确构造`layer_type="full_attention"`，
  源码指纹及片段见`mtp-layer-source.json`。优先检查草稿Attention/KV元数据；
  后续核对实际attention group及target GDN回滚索引，不假定draft自身运行GDN。
- 诊断涉及CPU拷贝/同步，必须与本轮无探针性能结果分开；诊断通过后移除探针
  再跑同协议性能及112K门禁，不能把带同步结果当默认收益。

## 恢复状态

原`qwen36-orangepi`已恢复，`state.txt=completed-restored`；health200、
短题8/8、A/B/A一致且无canary泄漏，唯一运行容器为该稳定服务。
见`restored-status.json`及`restored-mtp1-quality.json`。

本轮不新增部署补丁，不改变权重与默认参数。容器已停止保留，服务日志
`service.log.gz`、原始响应、逐位置计数、配置核对和汇总均保留。

日志中的一处`EngineDeadError`出现在主动停止容器的shutdown阶段；六条测量与测后
质量检查均在此前完成。未在本轮日志发现507错误或Aicpu kernel execute failed，
这不排除未被日志捕获的问题。
