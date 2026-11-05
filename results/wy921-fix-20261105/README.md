# WY 原生实现修复与完整门禁（wy-4）

当前结论：**待定，暂不建议采用**。扩充 G0 已通过；模型门禁在 A′ 对 A 的既有 logprob 比较步骤因 32K 会议输入首位置缺失 A 目标的 top-20 项而停止，B 未启动，不能建议采用。截止时间为 2026-11-06 01:00 UTC（09:00 CST）。

## 固定口径

阈值协议在查看修复结果前由提交 `04f0fb0` 冻结，SHA256：`e6327d44945f392f00b8e13897c7d48344cb5f9ecd2968c192b130b02458d93f`。逐文件输入哈希及协议全文在 adjudication.json。G1 复用 wy-2 冻结的 35 条 token 输入，SHA256：`e8fa2ed60bb717b2cad731988a0f85ba9dd07f39f0e614b2fb2e9b333b9e8c13`。

G0 包含原 30 项完整 GDN 用例、120 次真实 WY 调用快照，以及 12 个触发组合（每设备 6 个，v×1/2，g×1/0.25/0.0625，保留相关 k）。WY 的 q/k 必须精确，其余输出 atol=5e-4、rtol=0.04、relative RMSE≤0.02；完整 GDN atol=2e-4，其余相同。全部有限且三次逐位重复；候选逐点全部通过，132 个 WY 输入每个输出的总平方误差及超差数还必须不劣于同轮 Torch。平方误差计算底仅为 1e-12。

G1 设计要求同轮重跑 A/A′，八项阈值等于 A′ 对 A 的噪声底（倍率 1.0、无绝对豁免），在 B 前落盘冻结并记录 SHA；本轮因比较步骤中止，实际阈值未形成。固定栈长输入 prompt_logprobs 在 wy-2 已发生内存不足，本轮沿用获准的生成 logprob 退路：共同且相同条件前缀包含首分歧位、共享显式 top-20 支持上归一化 KL，并报告 greedy 首分歧。它不是全词表 KL，也不是完整 256 位 teacher forcing。

G2 同轮 P/A/B：GSM64、13 例结构化字段 C1/C3、8 短题前后，各计数差绝对值≤1。生产参数固定 TP2、MTP0、Graph[1,2]、maxlen262144、batch4096、seqs4、prefix on。现役服务配置不充当 P 臂。

性能仅在全部门禁通过后运行 P-A-B-A；会议回放耗时 B≤同轮较慢 A，输出工作量必须相同，并报告 A/A 漂移。旧栈收益不迁移。

## 修复与证据

只修改固定 PR17640 `18163f6b1d208ab3b4fb06e9752225611c37f40e` 的一个 WY 内核头文件；配套 H/O、gating、引擎、驱动和 CANN 均沿用原栈。修复有三处：

1. βK 在进入 Gram Cube 前被单次 FP16 舍入，扰动三角系统；把舍入残差拆为第二个 FP16 乘数补回。CPU 隔离复现中单 FP16 有 1 个 u 超差点，残差分解后为 0。
2. WY 内 g 的串行 FP32 前缀和改为补偿求和。它改变 WY 的累积计算，不改 gating。真实输入 g 总平方误差由约 3.81023e-4 降到 1.57528e-6，Torch 为 6.30727e-5。
3. 去掉经验性的 Frobenius 快速倍增准入；保留原有最大绝对行和≤0.75 的准入及既有 FP32 求逆路径。剩余两个真实超差点分别落在行和约 3.552 和 2.172 的额外准入分支；移除后消失。证据定位到该计算路径，未把其中 T 舍入和 βV 舍入进一步单独归因。

| G0 固定回归 | 同轮 Torch 对 CPU FP32 | 修复后原生对 CPU FP32 |
|---|---:|---:|
| 真实快照 | 98/120 | 120/120 |
| 最小触发组合 | 0/12 | 12/12 |
| 原完整 GDN 合成门禁 | 30/30 | 30/30 |
| 合计 | 128/162 | **162/162** |

132 个 WY 输入全部三次逐位一致；另加的三次连续排队、统一读回按字节比较也为 **132/132**。q/k 精确、所有输出有限、没有 Torch 回退。w/u/g 总平方误差分别为 4.68624e-4 / 7.70120e-2 / 1.57528e-6，Torch 为 2.13444e-3 / 1.64290e-1 / 6.30727e-5；全部不劣于对照，候选超差元素均为 0。

消融：只修 Gram 为真实118/120且 g 汇总误差不合格；再修前缀和仍118/120但汇总误差合格；再移除额外准入后162/162全部通过。候选源码 SHA256：`3ec7d34f7b3f43e142e69ecdb4e2dfaf0c5f9dd469b5682b8d3d3fe6b3e2dda8`；制品清单 SHA256：`8ff33c4e6f54efdc778952b804cd26b36e3d559c1e4653292a3bd1b45ca3d329`，模型控制器强制绑定此清单。

## 模型门禁和性能

A 与新进程 A′ 均完成 G1 35/35 条、每条256 token，共各8960 token；manifest逐字节相同，prefix-cache hits均为0。P、A 已完成 G2。B 尚未启动。

控制器打包带入归档旧比较器，遇到 `meeting-32768-0` 第0位的 A 目标 token 不在 A′ top-20 中时抛出 `ValueError: Missing A target logprob`。这是控制器打包疏漏，并非原生 B 的数值失败。按照“计划外问题写入汇报后停下”的约束，停止设备实验；未冻结实际噪声阈值、未运行 B、未测性能。当前仓库比较器仅用于本机离线诊断，保留截尾下界，不替代门禁判定；两个比较器 SHA 见 adjudication.json。

| G1 指标 | A′ vs A（离线诊断） | B vs A | 实际阈值 / 判定 |
|---|---:|---:|---|
| top-1 不一致率 | 0.002044989775（14/6846） | 未运行 | 未冻结 / 未判定 |
| 共享显式 top-k 条件 KL 均值 | 0.000887040430 | 未运行 | 未冻结 / 未判定 |
| 共享显式 top-k 条件 KL 最大值 | 2.371900279 | 未运行 | 未冻结 / 未判定 |
| 最大目标 logprob 绝对差 | **≥7.310889304**（1处截尾） | 未运行 | 未冻结 / 未判定 |
| top-1 首分歧后缀比例，均值 / 最大 | 0.2375 / 1.0 | 未运行 | 未冻结 / 未判定 |
| greedy 首分歧后缀比例，均值 / 最大 | 0.2375 / 1.0 | 未运行 | 未冻结 / 未判定 |

A′ 的 greedy 有14/35条与A分歧，最早位置为0（32K会议补充输入）；21/35条全256 token相同。无截尾用例中的最大目标 logprob 差为1.406968683。没有把32K用例的7.310889下界当作真实最大值或完整噪声阈值。

| G2 项目 | P | A | B | B 对 A/P 判定 |
|---|---:|---:|---:|---|
| GSM64 | 63/64 | 63/64 | 未运行 | 未判定 |
| 结构化 C1 | 13/13 | 13/13 | 未运行 | 未判定 |
| 结构化 C3 | 13/13 | 13/13 | 未运行 | 未判定 |
| 短题前 | 8/8 | 8/8 | 未运行 | 未判定 |
| 短题后 | 8/8 | 8/8 | 未运行 | 未判定 |

| 负载画像 | 本轮收益 |
|---|---|
| 会议26请求冷缓存回放 | 未测，G1/G2前置未完成 |
| RAG 8K/16K/32K前缀复用 | 未测 |
| 通用1K/4K/16K/32K×C1/C4 | 未测 |
| WY冷Prefill 1K/4K/16K/32K | 未测 |

B/A、B/P百分比及A/A性能漂移均无数据，不引用旧栈收益。

## 工具、补丁及验证

可复用工具：`tools/patch_pr17640_wy_fp32.py`（仅接受固定上游 blob，重复应用或源码不匹配直接拒绝），`tools/check_pr17640_wy_patch.py`（4 项 CPU 集成检查）。一次性控制器保留在本地 experimental/runs-20261105/，不提交。模型启动前核验发现控制器误用了旧观测类，已在首个 A 进程启动前换回既有数值门禁 worker；P 不加载该类，没有使用观测类产生的 A/B 数据。实际 worker SHA 和两个 A rank 的确认记录见 adjudication.json。构建在固定官方 9.2.1 镜像内用 CPU 和并发 2 完成，持本机重计算共享锁；未使用本机 GPU。

已执行命令（路径参数在运行时传入）：

```bash
python3 tools/patch_pr17640_wy_fp32.py --source-root "$SOURCE_ROOT"
python3 tools/check_pr17640_wy_patch.py --source-root "$PINNED_SOURCE_ROOT"
python3 -m unittest discover -s tools -p test_wy_snapshot_reference.py
python3 -m unittest discover -s tools -p test_compare_token_logprobs.py
# 固定镜像内，仅构建 WY 自定义包：
bash build.sh --pkg --ops=chunk_gated_delta_rule_compute_wy --soc=ascend310p --vendor_name=hyperqwen_wy4 -j2
# NPU 窗口由共享锁包住，窗口内恢复并等待独立 SSH 健康证明：
flock /tmp/a310-npu.lock python3 "$CONTROLLERS/locked-device-window.py" \
  --root "$RUNTIME_ROOT" --tools "$CONTROLLERS" --wy "$WY_CANDIDATE" \
  --triggers "$TRIGGERS" --captures "$CAPTURES" --out "$G0_OUTPUT"
python3 "$CONTROLLERS/analyze-g0.py" --captures "$CAPTURES" \
  --triggers "$TRIGGERS" --replay "$G0_OUTPUT/wy" --out "$CPU_ADJUDICATION"
flock /tmp/a310-npu.lock python3 "$CONTROLLERS/run-model-gates.py" \
  --root "$RUNTIME_ROOT" --tools "$CONTROLLERS" --wy "$WY_CANDIDATE" \
  --out "$MODEL_OUTPUT" --fixtures "$FIXTURES" --dataset "$GSM_DATASET"
```

CPU 固定补丁集成检查 **4/4**，独立 FP32 参考单测 **3/3**；本机离线诊断所用既有 logprob 比较器单测 **7/7**。设备门禁脚本 replay-g0-wy.py、原有完整 GDN 验证和本机 analyze-g0.py 的原始命令及日志保留在 raw/；162/162 的 CPU 裁定结果与候选清单绑定。

上游独立补丁：`results/wy921-fix-20261105/pr17640-wy-fp32.patch`，**5280 字节**，SHA256：`aa00aedca70ee286305df7580a05e793e62b5f7da1e0a8fcb3ba1f25f7135adb`。`git apply --check` 已通过；未发送或发布。原始补丁按结果目录策略留本地，可由可复用工具对固定基线重建。

## 恢复与真实限制

三个设备窗口均持有共享 NPU 锁；设备计算窗口约275秒、267秒和3111秒，均小于3小时。分别于2026-11-05 01:08:29、01:25:53、**02:42:51 UTC（10:42:51 CST）**，仍持锁时由新SSH独立确认原服务active、18081/health=200，随后放锁。最后窗口 finished 标记为02:42:52.459 UTC；采用先于此标记的首次健康记录作为恢复证据。稍后的重复健康检查不用于证明持锁恢复。

竞争服务在本轮始终未启动，恢复确认均为inactive且disabled。无运行中的本任务NPU窗口；共享资源已交还后续持锁线程。不操作交付站、本机GPU、生产镜像、CANN或驱动，不推送、不发布。

120次真实快照来自一个真实请求、两个TP rank，不能代表全部模型条件。扩充G0通过尚不证明旧轮全部模型偏差已修复。原30项完整GDN重复按既有零容差数值比较，新132组WY另做逐字节排队重复；未穷举跨流生命周期。

本轮A′已经出现32K输入首位置分歧及top-20截尾；后续需要先纠正控制器工具版本、明确截尾噪声的有效处理，再进入B。当前没有B模型门禁和性能证据。
