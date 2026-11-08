# WY / MTP 组合验证（2026-11-08）

初始组合诊断窗口仅在 a2 验证，不切换 a1、不推送；随后按用户镜像交付指令推送已验收的无MTP W8镜像，详见文末与registry登记。全程不改冻结 kernel、协议、规则、worker 或 R。镜像已有资产与本轮验证覆盖分别记录。

## 既有 TP1 镜像核查

已存在镜像：

```text
registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:8e17ec9ae646b1e418cf119bbb347238c4505b8b0037782a1345914824ec8d35
```

- 镜像 ID：`b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d`。
- `/opt/qwen36/start.sh` 设置 `QWEN36_GDN_ENABLED=1`、`QWEN36_GDN_LIBRARY=/custom/pr17640_gdn.so`、`QWEN36_GDN_SOURCE=/gdn-source.py`，并传入 MTP speculative-config。
- `/c8bootstrap/sitecustomize.py` 加载 `pr17640_gdn_adapter`。适配器将固定 PR wrapper 的 `_C_ascend` 替换为独立 `pr17640_gdn` namespace，映射 WY 为 `compute`，并同时替换 chunk 模块、消费者和 package 的导出。
- `/gdn-source.py` 的 chunk 前向调用 `_compute_kernel_inputs_from_npu_wy`；不满足能力条件时仍可回退 Torch。因此启动开关本身不能替代实际 native eligibility 证据。
- wrapper SHA256：`3b9bc0eb2e2e9d944316858df1efaea72776e7221b5eb1640db4a719b985accb`。
- 该 wrapper SHA 与本轮 TP2 锁定镜像构建清单一致；binding 命名空间与二进制不同，不能据此宣称整套运行时等价。
- WY 主源文件 `chunk_gated_delta_rule_compute_wy.cpp` SHA256 为 `3f42fad80232e02a77e3995cd64dbb7af010275698dc3123db629336cbfac0c4`，亦与 TP2 构建清单一致；这不构成全部 include 文件或 OPP 二进制一致的证明。
- binding SHA256：`7a8004d28f5f488f36a142b3ec453ea1df5ac8396e7244d787ffa10bfbb92047`。
- a2 已有环境配置为 TP1、W4A8、MTP2、long-prefill、C8 关闭；不能将 a2 的配置直接当作 a1 当前部署状态。

历史验证并非空白：`mindieq-gdn-experiments-20261013.md` 记录固定 PR GDN + W4 + MTP2 的 8K/32K 重复、GSM32 为 31/32、114560-token 检索，以及 MTP1/2 对照。4096 调度批量下 8K Prefill 为 503.099 tok/s、Decode 为 37.100 tok/s；这是历史配置记录，不能当作本轮同条件 A/B，也不能把完整 GDN 收益单独归给 WY。

此前“W4A8 + WY + MTP 未测试”的笼统表述应修正为：历史路径与镜像已存在；缺少当前交付资产对应关系、实际 native 路径确认及本轮组合对照。

## 本轮 TP2 W8A8 首次结果

使用锁定候选 `localhost/qwen36-pr17640-wy-locked:20261108`，image ID `9df3c6f3e42084d77ef258853cf824a011b4d30a1b05acc72a834a691c073163`。

在原 TP2 manifest 上仅增加 MTP1，保留 FULL_DECODE_ONLY、capture `[1,2]`、prefix cache 和其余设置。主模型和草稿模型加载成功，但模型图捕获失败。调用栈指向：

```text
_310p/attention/attention_v1.py -> forward_chunked_prefill_310
_310p/attention/attention_mask.py:95 -> query_start_loc.to("cpu", dtype=torch.int32)
aclrtMemcpy 107030 / captured-stream synchronization 107027
```

本实验使用 production manifest 启动覆盖入口；日志解析的编译模式为 `VLLM_COMPILE(3)`。镜像默认 `start-wy.sh` 的编译模式参数为 `0`，且尚未提供 MTP 开关。因此本失败只覆盖归档 manifest，不能外推为所有 Graph 配置均不支持 MTP。TP1 另带 `direct_graph_310` 兼容适配器，不能仅比较 Graph 模式名称而忽略运行时差异。

这是图捕获兼容性失败；没有请求质量或组合性能结果，不能归因为 WY 精度失败。实验容器已停止、日志保存、删除，独立快照确认 NPU 无运行进程。

结果目录（a2）：`/home/ychy/ascend-lab-runtime/hyperqwen-20261103/wy-mtp-matrix-20261108-v1/`。总耗时 231.316 秒。下一步仅关闭图执行做 MTP1 smoke 隔离；即使 eager 通过，也不能证明原 Graph 组合可交付。

## TP2 eager 最小隔离结果

同一镜像、同一模型和原 prefix 设置，仅移除 compilation-config 并增加 `--enforce-eager`。MTP1 启动成功，三条生成分别返回 `OK.`、`42` 和包含 `57.80` / `2026-11-08` 的合法 JSON，均正常 stop。耗时 217.445 秒；独立快照清场通过。

结果目录（a2）：`/home/ychy/ascend-lab-runtime/hyperqwen-20261103/wy-mtp-eager-smoke-20261108-v1/`；最小结果已归档至仓库 `results/wy-mtp-integration-20261108/tp2-eager-mtp1/`。这是基础可运行证据，没有完成 PDF、长输出或正式质量验收，不支持 eager 生产放行或性能收益声明。原 Graph 启动失败结论不变。

## TP1 本轮 native / MTP2 文本结果

既有8e17ec9a…镜像，TP1、C8关闭、MindIEQ long-prefill、MTP2、prefix关闭，编译mode 0及既有direct-graph适配器。日志确认 `native_eligible=True`，native臂若不满足能力条件会报错，不能静默回退。Graph捕获完成，三条基础生成均正确。

| 输入token | Prefill中位 tok/s | 有效重复 |
| ---: | ---: | ---: |
| 4096 | 635.552 | 3/3 |
| 8192 | 622.833 | 3/3 |
| 16384 | 595.862 | 3/3 |
| 32768 | 544.377 | 3/3 |

4K输入、128输出的Decode中位为35.637 tok/s，三次MTP草稿token接受率均83.33%。全部冷输入prefix hits为0。

v1窗口在PDF客户端读取输出目录变量时因 `PDF_AB_OUT` 未设置而中断，尚未发PDF请求。总耗时884.506秒，driver非零退出并独立清场通过。这是实验脚本错误，不是模型/PDF失败，也不构成完整矩阵通过。原始文本结果、日志与清场归档于 `results/wy-mtp-integration-20261108/tp1-native-mtp2-v1/`。

修复客户端变量后开启v2窗口，第一臂以SHA登记复用v1已完成的文本证据，续跑4页PDF、各2次，8/8满足脚本功能检查；其中pdf-01两次均达到512输出token上限，其余正常stop。这是功能smoke，不是完整OCR质量验收。prefix hits为0，但多模态缓存未禁用，不能将重复PDF时延称为完全冷视觉Prefill。结果归档于 `results/wy-mtp-integration-20261108/tp1-native-mtp2-v2/`。

随后按用户收敛范围的要求停止剩余W4对照，保留既有稳定profile与镜像。Torch/MTP2臂被主动停止，容器ExitCode 137属于取消清场，不计为模型失败；driver与本地自动收集器均已停止。独立检查确认无实验容器、NPU无运行进程。取消证据归档于 `results/wy-mtp-integration-20261108/tp1-scope-cancellation/`。没有完成当前Torch WY对照，不计算当前WY收益。

## 收敛后的交付范围（2026-11-08）

1. TP1 W4A8：复用现有稳定profile、原镜像digest和历史WY/MTP证据，停止额外的Torch/native × MTP0/2实验矩阵。本轮文本及PDF仅作为补充记录。
2. TP2 W8A8：沿用可复用的WY实现与Graph/MTP配置思路，保留W8量化入口、双卡映射及相应显存配置。先处理已复现的Graph启动兼容性；不将W4专用C8/MindIEQ扩展整体复制进W8运行时。
3. W8仅补交付必需的启动、实际native命中、Graph/MTP工作状态、代表性冷Prefill和PDF smoke；发生明确异常时再扩大排查。未经验证不将eager或MTP设为新的交付默认。
4. 验收后的W8镜像推送到既有registry的独立版本tag，并登记registry digest、image ID、base/source/binding版本、profile、启动命令和验收证据。W4已在registry，以既有digest登记即可，无需重建；确需统一别名时再核对别名对应digest。
5. 镜像登记与现有a1生产切换分别处理；本轮不触碰a1。冻结协议/规则/worker/R/kernel SHA均未变。原W8无MTP候选与未通过Graph验收的MTP扩展必须分别标注。

## 镜像身份记录修正（2026-11-08）

此前文档把TP1的image ID误记为86e64c35…；a2的v1/v2原始freeze.json、当前本地inspect及registry远端config均为b381d8e5…（完整值见上方与registry登记）。registry digest 8e17ec9a…未变，未更换运行镜像或冻结制品。此次修正只针对记录，保留原始实验快照。

## W8交付收尾：Graph / MTP2边界（2026-11-08）

在同一9df3c6f3…锁定镜像、同一W8权重上，固定TP2、mode 0、FULL_DECODE_ONLY、capture [1,2,3]、batch2048、seq1、max114688、prefix关闭，复用TP1的direct_graph_310兼容钩子，未改WY kernel。两次窗口结果：

| 窗口 | MTP配置 | 耗时秒 | 结果 |
| --- | --- | ---: | --- |
| w8-direct-graph-20261108-v1 | MTP2，主模型和草稿Graph | 265.338 | 启动捕获失败，未发请求 |
| w8-draft-eager-20261108-v1 | MTP2，主模型Graph、草稿enforce_eager=true | 267.974 | 启动捕获失败，未发请求 |

两次调用栈均从GPUModelRunner.capture_model进入主模型forward，在310P attention的get_splitfuse_mask调用query_start_loc.to("cpu")时触发107030/107027。第二次日志不再出现草稿ACLGraphWrapper包装，确认配置生效；失败依旧发生在主模型捕获，不能归为仅草稿图兼容性，也不能归为WY数学精度问题。容器ExitCode均1；两次独立清场均通过。没有运行冷Prefill、Decode或PDF请求，不形成新的组合性能/质量证据。

归档目录为`results/wy-mtp-integration-20261108/tp2-direct-graph-mtp2/`和`tp2-main-graph-draft-eager-mtp2/`；含启动manifest、source/image freeze、状态、清场与压缩日志。脚本及实验bootstrap归档在`experimental/wy-mtp-20261108/`。这些钩子仅挂载于诊断容器，没有加入已发布镜像。

本次交付收束为：TP1 W4A8复用既有WY/MTP2稳定profile；TP2 W8A8发布已验收WY无MTP镜像。W8 MTP Graph作为独立兼容性问题保留，不将eager实验通过或W4成功外推为W8组合验收。后续修复应优先检查主模型多token验证进入SplitFuse时的图兼容性；不为此改WY数学路径、放宽冻结门槛或拖延已合格WY镜像登记。

registry推送、digest拉取与远端manifest/config核验均完成；a2最终独立检查无容器和NPU进程，临时registry凭据文件已删除。a1未触碰。协议/规则/worker/R/kernel SHA未变，只有诊断配置和记录新增。
