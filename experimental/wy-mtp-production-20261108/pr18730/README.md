# PR18730 与本轮 Graph/MTP2 故障隔离

仅a2实验；不用于部署。复用2026-10-14的`opp-review`，其recurrent host/kernel源码SHA与本地PR分支`50fbb5a95`一致，含64位状态索引修正。OPP文件逐项摘要见结果`pr18730-candidate-context/artifact-sha256.json`。未重新声称本轮编译了kernel。

`candidate-gdn_310.py`保留当前镜像其余函数，只将`npu_recurrent_gated_delta_rule_310`函数替换为PR中的实现。原wrapper、PR wrapper与候选SHA见`wrapper-provenance.json`。需与修复OPP配套，不能只替换Python。候选图像本地ID：`e5497645fa3a6cae1dfa75d230d2e2efc5f5539f565ced2b9e75192fe564c736`。

`run_original_repro.py`在设备锁和空闲检查后运行旧缺陷的确定性复现。退出码0表示合法对照正确、旧缺陷出现、gather诊断正确三个断言均成立，不表示旧算子正确。

`run_candidate_regression.py`依次验证eager与Graph的42项状态/输出用例，包含变长、零长度、padding、legacy和完整状态行。源码只做语法检查；设备运行日志与命令归档在`results/wy-mtp-production-20261108/graph-localize/`。

实际运行时首次归档runner的`script_sha256`指向旧复现脚本，即使candidate实际运行的是check脚本；该字段不用于候选输入身份。以`command.json`和审计目录的test源码SHA为准；本目录后续runner已改为同时记录两个测试输入SHA。运行中的runner文件会快照，修改runner须等待窗口结束。

模型probe使用`run_profile.py --mode probe --seqs 2 --mtp 2 --eager 0 --entry ... --image ...`，保留显式环境变量与候选OPP入口。主模型Graph、draft eager；不扩长上下文性能验证。首次模型启动在CPU AVX2填充路径段错误，未进入模型请求，不算507015复现；后续入口增加ATEN_CPU_CAPABILITY=default，单列归档。

正确的OPP搜索顺序仍不足以修复模型Graph/MTP2 Mul故障。全局ASCEND_LAUNCH_BLOCKING=1被ACL Graph配置校验拒绝，未绕过。draft_sync_probe.py跳过capture，在eager draft逐算子同步；必须同时patch基类和310P子类，因为310P初始化会复制方法。仅patch子类的v1窗口无enter标记，其通过不能归因同步。有效v2在draft入口同步就失败，当前draft内部尚未执行。graph_boundary_probe.py追加主图前后、sample和proposal入口边界；prepare_sync_probe.py继续对eager _propose准备逐算子同步。所有探针仅定位，不用于性能或生产。

`prepare_sync_probe.py`定位到`attn_update_stack_num_spec_norm`里shape[2] int32 block IDs乘128。`repro_slot_mul.py`和`run_slot_cases.py`在独立进程11项算术均通过，不能写成通用CANN Mul缺陷。`patch_slot_index_dtype.py`只在slot地址计算前将block IDs提升为int64；`slot-index.patch`为实际变更。无探针候选seq2完整probe已通过。

原`start-opp-order.sh`来自不支持draft开关的旧入口，写死enforce_eager=true；即使runner传入DRAFT_ENFORCE_EAGER=0，也不能验证draft Graph。该seq4窗口仅作主Graph/draft eager证据。`start-opp-order-v2.sh`基于当前入口保留draft开关并加配套OPP顺序。runner新增`runtime-cli.json`核验容器PID1实际参数；非packaged模式下图开关、seq及MTP配置不符即停止验收。packaged仍单独归档实际参数，不能用外部实验参数标签代表镜像默认值。

单档[12]双Graph技术probe通过，但decode两窗约29 tok/s，主Graph三档/draft eager约46 tok/s，Prefill相同；筛查后主动停止额外双Graph容量窗口，不将中断算作模型fail。正式候选改为start-main-graph-release.sh。main-graph-release-full与同镜像packaged由run_main_graph_closure.py串行验收。角色审计已改为检查runnable名称；旧版use_eagle标签无效。详见docs/qwen36/wy-mtp-graph-localization-20261108.md。
