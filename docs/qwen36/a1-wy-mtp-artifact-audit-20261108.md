# a1 WY/MTP2 制品一致性只读审计

2026-11-08，读取本地验收清单及 a1 当前容器、21:35—21:42 journal；未发推理请求、未停止或重启服务、未操作 n2。

## 结论

a1 当前 staging 镜像不是 a2 完整验收的镜像，且缺少已验收的关键修复。确认存在生产化制品与证据绑定缺口；不能先将失败归为驱动差异或 WY 数学问题。原 a2 性能和容量证据只属于原 image ID 与配置。

| 项目 | 原 a2 验收制品 | a1 当前 staging 制品 |
| --- | --- | --- |
| registry digest | `a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5` | `777611b2e13da9be433e2e98ce67777713d07fc3321717af16a414e2594a5189` |
| image ID | `bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4` | `bb033d6b18c6f6ee342f13b1457f5c8cdfeeb90691698b3e68fa16f5535db952` |
| proposer SHA256 | `7d4664e9560de195dba043c0221b55757057993b2c6d2f8ffb955fcbad51e423` | `d7a992ca504eb23ff601b9a5654eb280879ed54e1507bfe38ca27690d83efbaa` |
| GDN wrapper SHA256 | `45bc54267d44d59e6b530de9b84ec2548c377de3abacf1688f29b9c7161d5fdc` | `1282ed500116e01a31db76ac0c6e44dbb8fd4d476460a43c2c07512733c32548` |
| `/pr18730-opp` | 已打包，SHA 清单登记 | 当前容器不存在 |
| prefix cache | 完整验收关闭 | 失败窗口开启，恢复后的诊断配置关闭 |

现场 proposer 代码仍直接计算 `block_ids * block_size`，没有原验收 int64 提升。仓库 deploy 入口与原 `start-main-graph-release.sh` 相比，也缺少 `/pr18730-opp` 的 OPP/动态库路径，且默认 eager/prefix 配置不同。仅匹配入口 SHA 和 CLI 不足以验证制品等价。

## 故障证据及边界

21:39:42 +08:00 两 rank 均报告 `Mul_9342884c99c006e21fc9d5d9260b8bef_high_performance_210000000` AICore 故障及 507015；随后 watchdog/EngineDeadError。Python 堆栈在 MTP draft attention 的 `get_splitfuse_mask` 内 `query_start_loc.to("cpu")` 同步时暴露异常。异步错误不能凭该堆栈直接确定 Mul 的发起位置；缺少旧 slot 修复与相同 Mul 症状高度相关，但仍需修复完整制品后的单请求验证才能证明因果。

只读检查时 systemd active 且 `/v1/models=200`，未验证当前诊断配置生成成功，不宣称业务恢复。a1 驱动 26.0.rc1 与原 a2 26.1.1 的差异仍是后续变量，当前不据此更换驱动或 CANN。

## 后续顺序

1. 先核验原 `a11b8aca…` 远端 manifest/config 与完整 build-manifest SHA，优先直接复用原制品；若重建，必须包含配套 recurrent OPP/wrapper、proposer int64、SplitFuse 适配器和原入口。
2. 新制品完整 SHA 与实际 CLI 核验通过后，在 a2 以原已验收配置（prefix off、主图[3,6,12]、draft eager）复测单请求与并发。不可把旧指标移到重建制品。
3. a1 维护窗口运行相同制品和配置 canary；若仍失败，再独立隔离 Graph/MTP/驱动环境。prefix on 另作功能与质量回归，不与制品修复同时引入。
4. 成功后重新测速度、更新部署模板与不可变身份。当前 `777611…` 保留为失败 staging，不能以 validated 状态放行。Quadlet、manifest、容器实际镜像和回滚入口必须成套验证。

原始验收依据：`results/wy-mtp-production-20261108/build-manifest.json`；现场详细日志读取副本位于本机 `/tmp/wy-a1-journal-audit.txt`（临时排查副本，a1 journal 为现场来源）。本轮没有修改冻结协议/规则/worker/R、WY 数学、设备运行配置或镜像。
