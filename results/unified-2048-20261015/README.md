# 三主机统一2048与MTP2验收（2026-10-15）

目标为TP1/W4A8/C16/MTP2/Decode Graph/GDN/MindIEQ E5、2048/512、max_model_len114688、seq1。用户已授权三台实验机重建。2048的选择来自a2对照速度接近，用于收束配置；未证明它更快或峰值内存更低。

## 当前实际状态

| 主机 | 当前实际配置 | 验收与例外 | 原始目录 |
| --- | --- | --- | --- |
| a2 EP310P3 | C16/MTP2/Graph/MindIEQ，2048/512，114688，util.95，async，prefix off，18024 | 本轮完整门禁通过 | /home/ychy/ascend-lab-runtime/unified-2048-20261015 |
| clt EP310P1 | 已恢复C16/MTP1/Graph/MindIEQ，2048/512，114688，util.90，sync，prefix off，18016 | MTP2未通过；恢复后短题8/8及A/B/A已通过；原同配置112000+128已通过 | /home/clt/qwen36-unified-2048-20261015 |
| orangepi RC310P1 | 已恢复C16/MTP1/Graph/MindIEQ，2048/512，114688，3GiB KV，sync，prefix on，18024 | MTP2未通过；MTP1恢复8/8、A/B/A通过，新2048长验收进行中 | /data/qwen36-deploy/results/unified-2048-20261015 |

当前共同核心为C16/Graph/GDN/MindIEQ/512/112K，三台本轮候选均设batch2048；MTP必须保留一项显式覆盖：a2=2，clt/RC=1。它是当前软件路径的验收差异，不等于310P1硬件永久不支持MTP2。RC是约87.5GiB统一内存设备，本轮不是物理48G RC或32G验证。保留原镜像、权重、算子和硬件兼容层，没有更换CANN。

统一公开入口仍为long-prefill，默认MTP2；需要当前MTP1基线时显式NUM_SPECULATIVE_TOKENS=1，自动选capture1,2，保留MindIEQ和2048/512。a2捕获1,2,3。不要为统一外观而开启未通过的MTP2。

## a2本轮完整验收

health → 8题+A/B/A → 32K warmup1/正式2 → 112000 warmup1/正式1（固定输出128、独立salt/prefix零命中）→ 112K跨段会议更新 → 长请求后8题+A/B/A，全部通过，见a2/validation/passed.json。

| 输入/输出 | Prefill tok/s | Decode tok/s | 样本与结果 |
| --- | ---: | ---: | --- |
| 32768/128 | 550.15 | 32.50 | 两次正式中位，接受率100% |
| 112000/128 | 384.35 | 21.70 | warmup后一次正式，接受率100%，prefix零命中 |
| 112000/54会议决议 | 349.45 | 21.36 | 预算180000、负责人周启、日期2026-11-12、延期上线，全部正确 |

阶段吞吐来自Prometheus request phase计数差，非纯kernel；合成样本不代表会议全量质量/P95。最大总序列配置114688，本轮完整固定输出请求验证112000+128；不宣称测到了绝对最大输入。长测后NPU快照39123/44277MB是采样占用，不是峰值或最小可部署预算。

## clt MTP2失败与局部修复的边界

1. 同步调度health200，首请求在rejection_greedy_sample_pytorch的repeat_interleave异常申请810428.12GiB，实际空闲约4.61GiB；不是合理容量需求。
2. 改async后同位置异常申请1000931.79GiB，未解决。
3. 源码把临时CPU草稿计数张量以non_blocking=True传NPU后立即用作repeat次数。独立speculative_metadata_310.py只改为blocking，原/改函数CPU 9组（含零草稿、不同拒绝位置）输出一致。真实请求不再巨量分配，但只通过5/8：384变3848、763变7636、NPU7329变NPU7PU732329；A/B/A仍一致。故局部修复不足以交付，不能只以“不再崩溃”通过。
4. 独立NPU sampler探针补source CANN后仍报INT64 ArgMinWithValue无受支持kernel；该探针没有完整复刻worker全部patch注册，不作为整模型根因证明。

clt已恢复原candidate.env、bootstrap、launcher及已验证MTP1容器；失败容器/日志保留。局部修复只保存在本轮results，不引入默认bootstrap、启动开关或新backend。

## RC MTP2失败与独立传输探针

- sync：health200，首请求持续无输出，py-spy停在_get_draft_token_ids_cpu → draft_token_ids_event.synchronize。
- async：普通文本会跳过上述草稿D2H路径，但首请求在_prepare_inputs复制num_computed_tokens时发生507018/aicpu exception，不能据异常上报位置认定复制是唯一根因。
- 独立新进程不加载模型，INT32/INT64 × blocking-default/async-default/async-copy-stream六个D2H复制事件用例全部通过（rc-copy-probe.txt，exit0）。因此不能说RC不支持异步/多stream复制；待定位MTP2/Graph/状态准备的组合协议。

当前恢复MTP1，同步调度、2048/512、C16，保留blocking MM H2D。新2048长测仍按完整协议执行，结果独立保存validation-mtp1，不借用历史4096结果冒充通过。

## 启动、追溯和回滚

- a2：qwen36-a2.service及dropin /home/ychy/.config/systemd/user/qwen36-a2.service.d/unified-2048.conf固定2048；实际运行源码5322528，模型/OPP未更换。启动前非强制rm仅清理停止容器记录，避免原恢复过程的占名重试；deployment.json已按实际inspect和本次passed.json更新。停止记录/日志先保存，未删除权重或挂载数据。
- clt：/home/clt/qwen36-optimized-20261015/start-candidate.sh与candidate.env已恢复原MTP1/2048；原同配置112K见handoff/clt-20261015。MTP2实验容器关闭restart并重命名保留，不同时初始化同颗芯片。
- RC：/data/qwen36-deploy/start-unified.sh → unified.env → run-service-unified.sh，现MTP1/C16/Graph/GDN/MindIEQ/2048/KV3GiB/ASYNC0/RC_BLOCKING_MM_H2D1。需重建容器，docker start不会更新参数。最初MTP1/4096基线容器名qwen36-orangepi-before-unified-20261015保留，需要回滚时先停新容器再恢复旧名，不并存初始化。

验收脚本默认完整协议；--smoke-only只生成smoke-passed.json，明确不表示新一轮长验收通过。后续优先修复310P1 MTP2的正确性和运行时阻塞，再讨论三台都启用MTP2；a2不继续无收益扫2048/4096；RC已有C8配对支持4096 Prefill约+11%，本轮C16/2048的32K约515.50，相对历史C16/4096的568.13低约9.3%，Decode约21.56/21.63基本持平。该C16比较不是交替A/B，须结合本轮112K后决定是否保留RC batch4096例外，不能把a2的batch结论直接套用RC。
