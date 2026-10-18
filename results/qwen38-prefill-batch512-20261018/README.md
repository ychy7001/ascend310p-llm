# Qwen3.8 TP2 / 16K Prefill batch512 对照（2026-10-18）

以[16K batch256 完整门禁](../qwen38-long-context-20261018/README.md)为前置，仅把 `max_num_batched_tokens` 和对应 `grouped_prefill_chunk_tokens` 从256提高到512，其他权重、量化、TP/EP、Graph、C16/MTP0、Prefix关闭、memory0.95与PG64全部保持一致。

## 为什么尝试

较长输入目前主要耗时在Prefill。原生W4 grouped路径把activation量化与路由展开分开，按grouped_chunk_tokens执行专家投影；256为首次启动时的保守值。提高到512有机会减少每份长输入需要的调度轮次，提高同批专家权重的复用。源码支持更大的显式边界，但不能据此断言更大必然更快或更准确，本轮只测512。

## 验证顺序

`run-after-baseline.py`等待batch256的16K计时、canary与交替请求全部通过，保存原始日志，停止但保留batch256容器。随后先执行 `run-b512-gdn-on-a2.sh`：

1. 两个rank，Qwen3.8 TP2本地HK8/HV24、128维、FP32状态，512-token GDN输出及最终状态对独立CPU递推；原始精确因果mask重复测试同时保留。
2. 两个rank，各4个原生W4A8 grouped投影用例：`K/N=2560/1280`及`640/2560`、总路由5120，分别覆盖稀疏局部专家和2560个本地路由、非对齐组边界、非本地输出清零。用固定社区测试中的独立CPU参考，真实group offset/scale协议，`rtol=.005, atol=.003`；这部分使用合成权重，不是整模精度分数。
3. 全部通过后才启动独立容器 `qwen38-tp2-text-16k-b512`，端口18038，同API模型名；重新profile，不使用显式KV预算跳过profile。
4. 完整模型：基本生成3题、8064/128和16256/128各一次预热+三次计时、16256 token中部口令检索、长请求后8条交替闭式答案检查。

运行时与模型路径见前置报告。本目录脚本复制到a2工作目录 `/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018`；额外工具为仓库 `tools/qwen38_gdn_tp2_probe.py`、`tools/qwen38_b512_native_gate.py`，以及长上下文报告列出的工具。

```bash
# 在batch256门禁正在执行或已完成后启动；会验证状态，失败不进入512服务。
python3 /home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018/run-after-baseline.py

# 如512候选不合格，回滚已保留的16K/batch256 Graph容器。
podman stop --time 20 qwen38-tp2-text-16k-b512
podman start qwen38-tp2-text-16k-graph
```

首次运行结果文件固定；重跑需新文件名/目录。已有原始数据不可覆盖。原始inspect环境快照只留a2。

## 结果与决定

独立算子门禁已经通过：两个rank的512-token GDN相对L2为0.0004644–0.0004648，最终状态最大绝对误差0.0003618–0.0005109，输出最大绝对误差低于0.000078；6次独立因果mask检查全部零误差。8个原生grouped投影用例全部通过，最大绝对误差0.0009765625、相对L2低于0.0000113，尾部非本地输出均为零。原始数据在 `b512-gdn/`、`b512-native/`，完整日志在 `b512-gdn-gate.log.gz`。

完整模型结果：8K/128 三次请求完成，TTFT中位约29.98s，Decode约15.98tok/s，较batch256的33.79s / 16.19tok/s，Prefill有效改善约11.3%，Decode下降约1.3%。但相同配置下16K/128请求长时间处于Waiting，未在合理时间获得调度；batch512的物理attention blocks由batch256的578降至504，启动日志显示峰值激活由约0.20GiB升到0.39GiB。

因此 batch512 不适合作为16K长上下文默认：它的8K Prefill收益不足以抵偿容量损失。当前已停止batch512并恢复16K/batch256 Graph服务；原始16K等待日志与结果文件保留在a2工作目录，正式报告会继续补充。
