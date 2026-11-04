# 上游反馈草稿：310P chunk GDN相同输入的输出不稳定

状态：本地草稿，未发布；根因未定位。复现仅使用合成张量，不需要模型权重或HTTP服务。

## 环境与最小复现

310P3，驱动26.1.1；torch2.10.0+cpu、torch-npu2.10.0.post2、CANN9.1.0-beta.1。vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，Ascend `88eac2713b14beedc7a48803274f8d6cee3e9f30`；生产chunk模块SHA `ab2b739bb6630ebe9fcb1d7af820f93adffce79668e32247bbb869119a0d6c69`。

使用[独立复现脚本](../../benchmarks/ascend310p/repro_chunk_gdn_nondeterminism.py)，在匹配环境启用custom operators：

```bash
python3 benchmarks/ascend310p/repro_chunk_gdn_nondeterminism.py --out original-repeatability.jsonl --repeats 48
```

固定seed311；B1/T111/Hk8/Hv16/K128/V128，FP16归一化Q/K、V/beta、非零初始state，FP32非正g。参考为独立CPU逐token递推，不复用WY/H/O。

## 预期与实际

预期输出和末state满足atol2e-4、rtol0.04、relativeRMSE<=0.02，并在相同输入重复调用时稳定。独立原路径一次运行仅5/12输出在容差内，偏差集中token56，末state逐位稳定；不同调用历史的结果分别保留。

完整链捕获中WY/FwdH输出与全部FwdO输入逐位稳定，raw FwdO输出仍偏离。冻结输入单独调用FwdO为48/48通过，原算法独立重编译也48/48；后者完整Torch-WY链18/24越界，同轮冻结WY-clone阶段0/24。同步、clone及分配历史会改变出现率，冻结单算子通过不足以证明完整链正确。

证据与跨运行栈对照见[Profiling](../ascend310p-profiling.md)，配套NZ状态契约和PR归属见[研究](../research/gdn-upstream-pr-evaluation.md)，9.2.1补充对照见[草稿](gdn-cann921-paired-validation-20261104.md)。本地候选与业务判定见[结果](../ascend310p-results.md)。面向上游的附件应含精确环境、脚本、原始JSONL及SHA，排除业务文本和权重。
