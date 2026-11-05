# WY921 第三轮：真实运行时偏差定位

结论：原生 WY 的逐点偏差已在单算子上复现，暂不修复；尚未证明这些小误差完整解释上一轮模型 logprob 差 ≥13。CANN compute-WY（B2）合成门禁 30/30，但真实 WY 快照门禁失败，停在 G1 之前。

本轮只运行一条 Torch WY 模型观测请求及离线算子对照，没有重跑原生 G1/G2、没有测性能。2026-11-04 13:21 UTC 开始，半天按 4 小时计，截止 17:21 UTC。固定官方镜像 `d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a`、CANN9.2.1、PR17640 `18163f6b1d208ab3b4fb06e9752225611c37f40e` 配套 H/O；TP2、MTP0、Graph[1,2]、maxlen262144、batch4096、seqs4、prefix on。观测容器返回原 Torch WY 张量，只增加边界采样同步。生产镜像和引擎算法均未修改。

## 1. 首次偏差

选择上一轮冻结集的 `replay-08`，765 输入 tokens；上一轮 B 的第一个生成 token 已与 A/A′ 分歧。本次只生成 1 token，得到 `101780`，与原 A/A′ 相同，原 B 为 `127326`。前缀命中为0。调度分成640和125 tokens，WY后者补齐到128。两 rank 各30层、每层2次，共120个快照，全部保留。

按既有 WY layout 容差，**两个 rank 都从零基 layer 0、各 rank 第1次 WY 调用的 `u_kernel` 开始超标**；当时 T=640、10 chunks、cu=[0,640]、初始 state 全零。rank0 Native↔Torch 的 u 最大绝对差0.015625，45个元素超差；Native↔CPU 最大0.00744343，6个元素超差。rank1同层同调用也超标。完整GDN算子对照的首个超标边界是该调用的末 state；全部120次 GDN 的 token output 本身均在既有完整GDN容差内。

这里的“首个”是同一组 Torch 运行时输入下，离线 WY/GDN 数值比较的首个边界；没有采集全模型每层 hidden states 来证明从该点到 logits 的因果传播。

## 2. 触发条件与原门禁覆盖

| 条件 | 原30项合成门禁 | 本轮真实/最小复现 | 判断 |
| --- | --- | --- | --- |
| 长度/chunk | T=1..1025，含64边界；最长补齐1088 | 640/128；缩到64仍超差 | 不依赖长序列或多chunk |
| varlen/batch | B=1，单序列及cu=[0,1,65,177] | cu=[0,640]、[0,125]；最小不需要cu | 多序列组合不是必要条件 |
| 初始state | 随机非零state | 首段零，后段非零；最小WY无state参数 | 非零state不是必要条件 |
| heads | HK/HV=8/16、16/32、8/24；D128 | 实际8/16；最小1/1，K=V=128 | 不依赖GQA或TP通信 |
| rank | 两个设备各15项 | 两rank首调用都超差；单设备单头复现 | 不是只发生在某一rank |
| dtype/layout/stride | q/k/v/beta FP16，g FP32；连续padding | 同dtype；输入全部连续、offset0；q/k/v format0，g/beta format2 | 不依赖特殊stride；元数据已留存 |
| q/k分布 | 独立归一化高斯随机 | k存在跨token相关性，最小最大点积0.999973 | 未覆盖真实相关分布 |
| v幅度 | `randn * 0.1` | 最小最大18.078125、RMS1.42574 | 未覆盖大幅度v与消去组合 |
| g衰减 | 均匀随机，decay=0.01/1/8 | 最小[-0.00611619,-0.0000570009]，chunk和−0.04549933 | 弱衰减、相关k、大v组合未覆盖 |
| beta | 均匀0.05..0.95 | 最小0.467773..0.963867 | 相关分布与上边缘未覆盖 |

最小失败保留原序列192..255、第14个value head（零基），缩至B1/T64/HK1/HV1/K128/V128，连续输入，无模型/state/varlen/TP依赖。保持其余输入不变，v缩小10倍或100倍，或g固定−1，或k改为随机归一化向量，Native↔CPU均回到容差内。beta固定0.5使Native↔CPU通过，但Native↔Torch仍有1个元素超差。

目前定位到混合精度/消去敏感的输入组合，未锁定唯一错误指令。原生与CANN核心代码的额外Frobenius快速分支不是这个最小失败的原因：该例范数平方109.1624，大于7.50218分支阈值。没有对内核作修复。

## 3. 三方离线复现

WY沿用原layout门禁：q/k精确，其余 `atol=5e-4, rtol=0.04, relative RMSE≤0.02`。完整GDN沿用30项门禁 `atol=2e-4, rtol=0.04, relative RMSE≤0.02`。协议在首次候选设备数据之前冻结，SHA见adjudication；没有因本轮结果放宽阈值。

CPU FP32 WY参考直接解单位下三角系统，独立于Torch的行更新；3项CPU测试覆盖直接token递推、原行更新交叉核对及局部误差拒绝。完整GDN参考复用现有独立CPU逐token递推。大解析均在本机，CPU线程≤2；远端设备任务串行。

| 对照 | WY在容差内/120 | 完整GDN output+state在容差内/120 |
| --- | ---: | ---: |
| Native↔CPU FP32 | 111 | 107 |
| Torch↔CPU FP32 | 98 | 104 |
| Native↔Torch | 100 | 111 |

Native/Torch重放全部有限；Native重复120/120逐位一致，Torch离线输出与模型截获120/120逐位一致；q/k布局精确。完整GDN的超差均在state，120个token output均在容差内。故不能根据“Native↔Torch不一致”单独认定原生结果更差，也未复现NaN、布局错位或运行间不稳定。

## 4. 最小复现包、修复与B2

复现包路径、大小、逐文件SHA见 `repro-manifest.json`。独立审查后重新打包：`upstream-gdn-wy921-runtime-repro.tar.gz` **279,469字节**；无权重、prompt、tokenizer或二进制，使用可无损还原的数值JSON输入与设备记录输出，含英文README、CPU参考/测试/校验、固定PR绑定构建及对照脚本。tar身份与时间元数据已清除。需使用CANN9.2.1和固定PR源码重新构建设备部分。发布状态见本文末尾审查记录。

最小例Native↔CPU的u最大绝对差0.0039176941，1个元素超差；Native↔Torch最大0.01171875，2个元素超差。Torch↔CPU也超差。原生内核未修改，故没有“修复后重新门禁”的结论；原生整模型留待后续轮决定。

B2使用固定ops-transformer `0340eae051ddf5ab956df0f7e0396dbd6314cfc5`及上一轮CPU构建制品；薄绑定仅注册 `cann_wy921::compute`，vendor为`wy921_cann_transformer`，不占用`cann_ops_transformer` namespace，unsupported直接报错，不回退。沿用原Torch gating，未接入FusedGdnGating。两设备各15项，**合成30/30通过**，每路径每例3次重复；实际映射确认CANN WY与配套H/O两个私有库。真实快照CANN↔CPU为113/120、CANN↔Torch为101/120，逐点门禁失败，因此B2 G1/G2均未运行，原35条输入和A/A′阈值文件未改变。

## 过程问题与限制

- 第一次原生设备重放在数值调用前因控制器覆盖PYTHONPATH缺`tbe`失败；保留原路径后完成。该失败不计数值门禁。
- B2首次合成因vendor初始化顺序错误报缺`_pattern`；同批独立WY快照重放完成。shell串接误将后续成功作为总退出码，已改为失败即停，该轮不计门禁通过。
- B2第二次15个数值用例通过，但映射检查发现H/O来自镜像内置库，拒绝计作配套H/O门禁；最终恢复bootstrap后的显式vendor顺序，两设备30项及来源检查均通过。中间失败原始日志保留。
- 仅截获一条765-token请求，未穷举冻结35条和32K运行；未证明本轮小数值差完整解释旧模型logprob大差。观测增加CPU同步，可能改变时序。快照保存了原shape/stride/format，离线重建为连续张量；连续最小输入仍可复现。
- 原始快照约1.015GB，只存本地忽略目录；结果目录只提交本README、adjudication.json和复现包清单。

工具与控制器提交：`f3fa439`（wy-1: 9）；仅本地提交。

## 恢复

每一阶段均在finally中停止实验容器并恢复原服务。最终于 **2026-11-04 14:04:23 UTC** 通过独立新SSH会话确认原服务active、18081/health=200；运行中的rootful容器仅原服务，rootless实验容器为空。总用时约43.1分钟，未超时间盒。`<local-image-service>` 全程保持停止，结束后也不启动；它仍为enabled。本轮未操作另一实验机，未更换CANN/驱动或宿主安装，未推送。


## 2026-11-05 独立审查与上游发布

- 逐项结论与真实限制见 `review-report.md`；归档逐文件审查见 `public-package-audit.json`。
- 修复原tar用户名元数据、二进制输入发布缺口、内部vendor名称与构建说明；所有公开数据均为数值JSON或文本脚本，无权重或业务文本。
- Python3.11.15 / torch2.5.1+cpu：三项参考测试、输入无损重建/回读、张量SHA/shape/dtype/有限性、记录输出重算均通过；新解压包重复验证通过。固定PR的两个Torch函数与历史观测函数AST一致，CPU执行通过。
- 本次未重新执行NPU构建/运行；设备数字来自2026-11-04保留输出，原精确驱动版本未保留。原先50,410字节二进制输入包由279,469字节纯文本包替代，SHA见清单。
- 创建公开gist后，重新下载12个文本文件，逐一核对SHA；发送前再确认PR OPEN/未合并/head精确匹配18163f6b。只发一条普通英文评论，无review或issue/PR操作。
- gist：https://gist.github.com/ychy7001/b94596c3949f19012d635f97b203dd23
- 评论：https://github.com/vllm-project/vllm-ascend/pull/17640#issuecomment-6646514137
- 评论正文见 `comment-body.md`；发布时间 2026-11-05T00:52:37Z，正文已与远端回读核对。本地提交，不推送。
