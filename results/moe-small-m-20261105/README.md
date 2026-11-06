# MoE 少 token tiling 实验（4b-①）

最新结论（2026-11-06）：接续五对回放完整完成，5/5均更快，中位墙钟下降1.534906%，未达到冻结3%门槛，不采用，不继续画像验收。历史0/5与冷启动失败记录保留。

**状态：桥接链接修复、G0、新同轮G2及双rank稳定Profile均通过；正式回放因新启动期native段错误按计划外问题规则停止。** 完成1个A臂、0/5完整配对；RAG/通用画像未运行。模型当前不采用，没有端到端收益或非劣结论。

本项仅改变MoE小M调度，不叠加QKVZ量化，不改权重、量化方式和其他算子。基线`74f7202`。A/B固定相同镜像ID、只读模型目录及TP2、seqs4、Graph[1,2]、MTP0、maxlen262144、batch4096、prefix on。候选在专家循环前计算全部输入行的动态scale并同步一次；逐专家GEMV沿用原始算术与行号。仅在routed M=8/16/24/32使用Torch桥接，分组、swiglu和通信保留。

桥接构建采用显式`LINKER:-rpath,$ORIGIN`，实际ELF含`RPATH=[$ORIGIN]`；设备库SHA保持不变，未添加桥接目录的运行时搜索路径补丁。新工具在全新Python进程导入Torch/Torch NPU后直接`torch.ops.load_library`桥接库，检查依赖未预加载、算子原先未注册。2026-11-05 **04:01:57 UTC**持锁检查通过；随后B整模型G2与Profile均成功运行。原装载失败属于已修复的构建遗漏。

G0：真实双rank四档M×两投影共144/144案例与官方和自编译对照逐位一致，432/432重复及144/144修改输入后的Graph replay核验通过。CPU FP32 rtol=0.002、atol=0.002，max_abs=0.0019521713，容差外0。按用户确认的逐位一致分支，G1免做。

| 微基准档位 | 计时 | 两投影合计加速 | 会议回放降幅贡献（推算） |
|---|---|---:|---:|
| B1 | Graph | 1.7139× | 3.8291% |
| B2 | Graph | 1.6048× | 1.2426% |
| B3 | eager | 1.6064× | 0.2657% |
| B4 | eager | 1.6401× | 1.1590% |

转移预算合计 **6.4965%**，达到3%进入整模型门槛。候选微基准通过实际Torch桥接调用，包含其分配、recordStream及launch开销；不是仅计设备函数。真实bank仅每rank首层，路由观测覆盖全部40层共2904条。先前固定stream导致空Graph的批次已整体作废；有效数据均有输入变更重放核验。

| 新同轮G2 | A | B | B−A |
|---|---:|---:|---:|
| GSM64 | 63 | 63 | 0 |
| facts16 | 16 | 16 | 0 |
| fields_C1 | 13 | 13 | 0 |
| fields_C3 | 13 | 13 | 0 |
| short_before | 8 | 8 | 0 |
| short_after | 8 | 8 | 0 |

G2各项减少均≤1，通过。26字段输入payload/tokenization/usage一致，解析值全部精确等于独立预期；16事实解析值一致；前后复用相同、无canary泄漏。GSM自由文本58/64相同，不能将算子逐位一致扩大为整模型文本逐位一致。

双rank原生Profile共16捕获（A/B×B1–B4×2rank），各112–113观测步；每捕获选96稳定步，目标kernel均7680次（80/步）。B1/B2为FULL Graph命中，B3/B4为eager。CANN在Graph记录候选名`moe_prescale_0`，eager为`moe_prescale`，工具按实际模式精确核验。

| 档位 | A MoE ms/步 | B MoE ms/步 | A整窗 ms/步 | B整窗 ms/步 |
|---|---:|---:|---:|---:|
| B1 | 7.457 | 4.678 | 35.778 | 32.947 |
| B2 | 12.123 | 8.005 | 45.499 | 41.577 |
| B3 | 17.470 | 10.720 | 64.249 | 76.041 |
| B4 | 22.318 | 13.192 | 68.722 | 78.955 |

这些计时带Profile：eager档目标kernel下降、整窗变慢。它们只解释执行，不能代替无Profile端到端裁决。原始捕获在设备侧不导出；本机用匹配CANN在共享CPU锁内串行解析/导出，禁止SDK子进程扇出，新建日志umask0027满足SDK权限检查；未修改样本字节。

正式回放次序预先冻结AB/BA/AB/BA/AB，各臂重建冷进程。首A臂05:20:52 UTC健康，启动210.002秒；05:31:37完成回放，墙钟 **644.948969秒**（不含启动/图捕获），11请求、200895输入、17045输出、prefix零命中，11831 scheduler/worker/joined steps全部对应，未匹配步0、阶段墙钟残差0。

首B臂约 **05:32:34 UTC** 日志打印`Segfault encountered`：native栈为CPU AVX2的FP16 `TensorIterator` fill/zero/zeros。另一worker打印small-M桥接安装成功，随后EngineCore初始化失败；05:32:44控制器检测容器退出。没有再次出现缺失`libmoe_prescale.so`错误。host kernel journal该时间段无记录，不能由此判定更深根因。没有B回放请求、没有完整配对；按计划外问题规则停止，不修改配置、不重试、不扩展排查。

| 负载画像 | 实测收益/完成度 | 判定 |
|---|---|---|
| 会议回放 | A单臂644.949s；完整配对0/5 | 无配对收益，未裁决≥3% |
| RAG 8K/16K/32K Prefill ×C1/C4 | 未运行 | 未验证非劣 |
| 通用1K/4K/16K/32K ×C1/C4 | 未运行 | 未验证非劣 |
| WY Prefill | 未运行；本项不修改WY | 无新增结论 |

门禁与工具验证命令（设备步骤须在独占flock窗口内执行，退出前恢复原服务并用新SSH确认health200）：

```sh
python3 tools/check_moe_bridge_load.py --library "$BUILD_DIR/libmoe_prescale_torch.so"
python3 tools/bench_moe_prescale_310.py --library "$BUILD_DIR/libmoe_prescale.so" --snapshots "$CAPTURE_DIR" --output "$MICRO_JSON"
python3 tools/eval_gsm8k_http.py --base-url "$BASE_URL" --model "$MODEL" --dataset "$GSM_DATASET" --limit 64 --seed 310 --max-tokens 512 --label "$ARM" --out "$GSM_OUT"
python3 tools/check_business_facts.py --base-url "$BASE_URL" --model "$MODEL" --cases 8 --concurrencies 1 3 --out "$FACTS_OUT"
python3 tools/check_action_state_regression.py --base-url "$BASE_URL" --fixtures "$ACTION_FIXTURES" --out "$FIELDS_OUT"
python3 tools/check_cache_request_reuse.py --base-url "$BASE_URL" --model "$MODEL" --out "$SHORT_OUT"
flock /tmp/a310-cpu-heavy.lock python3 tools/adjudicate_moe_stable_profiles.py --root "$LOCAL_PROFILES" --out "$LOCAL_PROFILES/native-adjudication.json"
python3 tools/replay_frozen_arrivals.py --base-url "$BASE_URL" --model "$MODEL" --plan "$FROZEN_PLAN" --out "$HTTP_EVENTS"
flock /tmp/a310-cpu-heavy.lock python3 tools/analyze_decode_observation.py --observation-dir "$OBSERVATION" --http-events "$HTTP_EVENTS" --out "$PHASE_ANALYSIS"
python3 -m unittest discover -s tools -p 'test_*moe*.py' -v
python3 -m unittest discover -s tools -p test_replay_frozen_arrivals.py -v
```

CPU裁决测试15/15、固定到达HTTP联调1/1，共16/16；持锁冷装载1/1；原生Profile16/16。完整回放采用裁决及14画像的输入/usage/prefix与Prefill/吞吐/TTFT/TPOT非劣工具已预先冻结，尚无完整模型测量结果。

只读服务journal核查：本控制器01:47:21 UTC启动原服务，01:50:45应用启动完成，01:50:46健康，01:51:00新SSH确认200，01:51:01放锁。同秒journal先记录外部`systemctl stop`与systemd Stopping，随后EngineCore收到SIGTERM，01:51:05为`Deactivated successfully`；与接续WY持锁窗口及控制器停服步骤相符，无自行崩溃证据。WY随后02:39:12启动恢复，02:42:46应用启动完成；图片实验接锁后02:43:00主动停服。

本轮G2+Profile窗口04:01:47–04:57:19 UTC，04:57:04恢复健康，04:57:18新SSH验活，共3331.466秒。本次正式回放窗口由本控制器恢复原服务：**05:36:09健康，05:36:12新SSH确认200，05:36:13放锁**，持锁 **1135.221905秒**。两窗均小于3小时，原配置恢复，未启动禁用的图片服务。全部后续自动控制器已退出，未申请其余实验窗口。

执行偏差：追加修复时一次挂载设备的纯装载探针漏共享锁，已向用户说明，该结果不计门禁；04:01:57持锁重做通过。所有raw日志、张量、启动栈与恢复证据留在忽略目录。

真实限制：首层bank微基准不覆盖所有层；144案例逐位一致不代表所有模型执行；Profile包含观测开销；正式B启动native崩溃根因未定，0配对不能判断端到端收益；RAG/通用/WY未测，人工会议语义审阅未做。未推送、未合并、未部署，未用其他候选补收益。


## 2026-11-06 启动问题接续审查

镜像线首次候选失败与本项首次正式 B 回放失败的原始日志，最前 14 个 native 栈帧完全一致，包含 AVX2 FP16 `TensorIterator fill/zero/zeros`。这只定位了共同的崩溃路径，不能证明共同根因，也不能据此声称 MoE 无关或故障已修复。

本机同镜像、单线程、纯 CPU 的 21 种长度×20 次 zeros/fill/zero 共420例通过，未复现崩溃。由于本机缺 Ascend 驱动，CPU 探针关闭了 NPU 后端自动加载；不等价于目标设备环境。最初直接导入 Torch 的探针仅因缺驱动库失败，不计崩溃证据。

后续按新窗口规则清场，不再执行旧脚本的恢复服务流程。正式回放前先做一次独立候选冷启动，失败即停止；原正式回放仍是0/5完整配对。数值与性能结论不变。

接续性能工具审查：回放墙钟/TTFT/TPOT和画像四项指标现在拒绝NaN、无穷、布尔值及非正数，避免无效分层延迟绕过退化检查；原五对顺序、3%/4对/分层3%门槛不变。回放与画像单测各5/5，历史完整A回放11请求、644.953276s通过新的有效性校验。新观察尚未采集，接续工具哈希在设备B前另行冻结；不覆写旧冻结计划或旧失败结论。

11-06接续冷B检查通过：双rank均记录候选worker安装，服务健康启动，ExitCode0、无OOM，窗口216.89s，清场后两服务inactive、两类容器与NPU进程为空。没有复现此前FP16初始化段错误，根因仍未定位；不能把这次启动成功写成修复。原五对AB/BA回放现已按2+2+1对分为三个独立持锁窗口串行采集，任一失败即停、不重试；完整后才判收益，目前没有新的端到端采用结论。控制器以独立会话运行并保存PID/逐窗口状态，聊天中断不再依赖终端存活。

## 2026-11-06 五对回放收口

十臂均11/11请求完成，共110请求、零错误；每臂ExitCode0、无OOM。五对墙钟下降依次1.520569%、1.468459%、1.676108%、1.542377%、1.534906%；中位1.534906%，5/5更快。足量分层的TTFT/TPOT未出现超过3%的退化。冻结的中位墙钟下降至少3%未达到，因此回放筛选不通过、不建议采用；不降低门槛、不追加重试，不再运行RAG/通用画像。

三个正式窗口依次3424.59、3412.95、1707.97秒，均正常清场；结束独立查看确认无实验容器、两个服务inactive、NPU锁释放。分析在本机持重任务锁完成，完整逐臂日志、阶段分析和窗口记录保留，SHA见adjudication.json。算子6.4965%预算不能替代本次实测1.534906%回放结果。RAG、通用与WY Prefill收益本次未测；原冷启动段错误根因仍未定位。
