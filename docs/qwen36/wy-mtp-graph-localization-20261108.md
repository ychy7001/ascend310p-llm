# 310P W8 WY/MTP2 Graph 定位与配置选择

2026-11-08，a2实验；a1只读。结论是保留native WY，生产候选选择主模型Graph `[3,6,12]`、draft eager。完整容量与同镜像封装入口工程验收已通过并发布；业务质量边界见生产配置报告。

## 三个独立问题

1. **PR18730状态索引缺陷确实存在。** 原镜像wrapper仍展平状态表，配套OPP缺少stateIndexStride。旧缺陷在原制品确定性复现；按源码SHA核对后复用旧opp-review配套构建，只替换recurrent函数，eager和Graph状态/输出回归各42项通过。该修复必须成套使用Python/host/kernel，不能只替换wrapper。复用构建不声称本轮重新编译。
2. **本轮Mul不是上述缺陷的同一复现。** 配套状态修复且正确注册OPP后，模型双请求仍触发相同507015。逐算子同步把首次失败定位到`attn_update_stack_num_spec_norm`的`block_ids * 128`：int32、shape[2]、stride[1]、offset0。当前draft前向还没执行；旧SplitFuse栈只是异步报错暴露点。独立进程11项算术均通过，尚未证明通用CANN Mul缺陷。单点兼容补丁在乘法前将block IDs转int64，保持slot地址公式；去除同步探针后，seq2和seq4主Graph/draft eager完整probe通过。
3. **draft Graph另有资源与性能取舍。** 真正开启draft Graph后，三档捕获在2/3时耗尽event资源。收敛为单档[12]后，seq4双Graph短probe通过；但小批次需补齐到最大档，decode连续两窗约29.317/28.792 tok/s，而主Graph三档/draft eager约46.430 tok/s；相同4K Prefill约2045 tok/s。这是顺序诊断窗口的筛查，尚非交替A/B，足以先不选择更慢的双Graph策略。

因此目前保留draft eager有实测理由。主模型图优化已经开启；WY数学路径、量化配置与CANN版本未改。新增性能比较不能代替以前的模型质量结论，也不能用来宣称新的WY加速比例。

## 证据边界

- `pr18730-seq4-mtp2-dual-graph`名称有误：旧入口写死draft eager。结果只算seq4主Graph/draft eager，不能算双Graph。
- `pr18730-seq4-mtp2-true-dual-graph`是真正双Graph三档启动失败；`pr18730-seq4-mtp2-dual-single-capture`是真正单档双Graph短probe通过。
- `graph-release-r1-full`在性能筛查后主动停止；容量请求不完整不能算通过，也不能归为模型功能失败。后续r2-packaged串行门禁已取消，没有新设备动作。
- 第一版生命周期审计用use_eagle识别角色不正确，因为主图也设置此字段。其主/draft标签无效；后续审计记录实际runnable名称，并分别验证capture/replay，不增加设备同步。
- runner现在核验PID1实际参数，非packaged窗口若主/draft图开关、seq或MTP设置与目标不符就停止。packaged入口SHA从镜像读取，不能以未挂载的host文件代替。

结构化证据见`results/wy-mtp-production-20261108/graph-localize/pr18730-decision.json`和`graph-policy-performance-screen.json`。独立状态回归见`pr18730-operator-state-audit.json`；逐算子首错见`pr18730-seq2-mtp2-prepare/first-failed-operation.json`。

## 当前收尾

`main-graph-release-full`使用镜像`localhost/qwen36-w8-wy-mtp-main-graph:20261108`（image ID `bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4`），目标为TP2/W8A8/WY/MTP2、262144上下文、seq4、batch2048、memory0.9、prefix关闭、async开启、主Graph三档、draft eager。

通过完整qualify和独立清场后，`run_main_graph_closure.py`才启动同一镜像的无源码挂载packaged验收，并关闭生命周期审计。它核对同image ID、同入口SHA、实际CLI与主图两rank的capture/replay。失败则取消后续动作；不切a1、不推送Git。

收尾结果：main-graph-release-full与main-graph-release-packaged均complete且独立清场通过；主图两rank capture/replay、同image ID与入口SHA已核对。registry digest为`sha256:a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5`，远端manifest/config核验通过。生产配置及限制见[满上下文部署配置](wy-mtp-production-20261108.md)。
