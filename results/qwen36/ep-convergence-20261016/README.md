# EP 参数收束复验（2026-10-16）

目的：在 clt EP310P1 上复验原 MTP1 长上下文基线，再测试独立 MTP2 兼容性候选；保留 a2 已通过的 MTP2 和 RC 已验证 MTP1/4096。按运行时组合验收，不根据 SSH 名称或 SoC 型号自动选择算法。

## 实验条件

- clt：x86_64、EP310P1、driver26.0.t1，可见44215MB；固定原镜像 image ID、W4/GDN/MindIEQ 制品、C16、同步调度、util.90、batch2048、expert chunk512、max114688、max_num_seqs1、prefix off、language-model-only。
- 基线：原容器 `qwen36-c16-112k-20261015`，MTP1、capture1,2。
- 候选：独立 `qwen36-ep-mtp2-slot-20261016`，MTP2、capture1,2,3；原容器停止保留，结束后自动恢复并复验。
- 候选额外两处局部修复：greedy sampler 的临时 CPU 草稿计数改成阻塞上传（不替换算法）；第二草稿 slot 商/余数采用有界 block128 设备计算。后者要求 int32、max_len<=1048576，拒绝未知形状。
- 这是一组 MTP2 兼容性候选验证，含两处局部修复，不能视为只改变 MTP 数量的纯性能 A/B，也不能单独归因哪个修复有效。
- clt 已有 accepted-count snapshot，与 RC 最初故障前源码条件不同；本轮不重复移植 snapshot 补丁。

协议：短题8/8与A/B/A → 32768/128（warmup1+正式2）→ 112000/128（warmup1+正式1）→ 112000会议四字段 → 长后短题。候选先通过短门禁才进入性能测试。每条冷请求独立 cache salt，检查 prefix hit=0。计时为独占 API 的 Prometheus 请求阶段耗时，Prefill=input/time，Decode=(output-1)/time，非纯 kernel 时间。会议输出长度不同，单独报告。

## 源码与制品核对

`clt-source-hashes.json` / `ychy-a2-source-hashes.json` 为只读核对。两台 EP 的 proposer、rejection_sampler、runner、patch_mamba_utils、direct_graph_310、W4/MindIEQ/GDN 扩展、GDN Python 源码和模型 config 均相同。初次记录的两台 image ID 不同；后续检查发现 Podman/Docker 使用不同标识，registry digest 与18层 RootFS 完全相同，不能据此推断基础镜像内容差异。纠正证据见 `../runtime-image-20261016/base-comparison.json`。本轮未比较全部权重或 OPP 树。

已检查文件中 bootstrap 差异见 `bootstrap-diff.patch`：clt 额外启用 `clt_sync_metadata`，可选加载 KV audit。其源码已在 `../clt-deployment-20261015/clt_sync_metadata.py` 归档。该差异只是排查范围，不证明因果。两台还有调度、视觉、利用率和 CPU 线程等实际配置差异，分别见 runtime JSON，不用跨主机绝对速度归因 SoC。

## 复现

现场目录 `/home/clt/qwen36-ep-convergence-20261016`。`run.py` 是本次顺序编排，固定容器名与路径，使用原容器 inspect 克隆，不是通用生产部署脚本。需独占18016和设备0；不能在现有目录再次运行，输出使用排他创建。

```bash
# 在新的实验目录放入本目录的脚本和 tools；调整 run.py ROOT/候选名称。
docker inspect qwen36-c16-112k-20261015 > baseline-inspect-original.json
python3 prepare_clone.py baseline-inspect-original.json baseline-inspect.json
docker cp qwen36-c16-112k-20261015:/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py proposer-original.py
python3 prepare_proposer.py proposer-original.py proposer-slot.py
python3 -u run.py > run.log 2>&1
```

`run.py` 显式使用 bash entrypoint，`prepare_clone.py` 去掉原 Config.Cmd 中重复的 bash，保留 -lc 和完整脚本。现场在克隆前做了相同输入规范化，原始 inspect 独立保留；模型参数不变，说明见 `clone-command-normalization.json`。

原 proposer SHA256：`4a90532c5c3667d9eb4fec666bf03c65a2f1c92cd402f731b644ef8fa2ed82ab`；候选：`aded1eb55b30ed43bb34a21b858810af6ddad801b2e33fef15e5acc134e6428b`。准备脚本按哈希拒绝未知源码和覆盖已有输出。仓库保留最小 patch，不重复提交整份上游 proposer；候选只读挂载，不改安装目录。失败候选不会进入默认部署入口。

## 结果与门禁决策

| clt MTP1 基线 | 正式样本 | Prefill tok/s | Decode tok/s | 结果 |
| --- | ---: | ---: | ---: | --- |
| 32768 输入 / 128 输出 | 2 | 568.93 / 570.10 | 24.51 / 25.16 | 均有效，prefix hit0 |
| 112000 输入 / 128 输出 | 1 | 417.49 | 16.61 | 有效，草稿接受64/64，prefix hit0 |
| 112000 会议输入 / 54 输出 | 1 | 378.29 | 15.56 | 四字段全部正确 |

每个固定长度组另有1次预热，不计入表内。32K中位数569.52/24.83。原服务前后短题均8/8，A/B/A一致，无canary泄漏；会议结果与重复填充输入分开，不能将不同输出长度混作吞吐 A/B。单个112K样本不是P95、峰值容量或完整业务质量证明。

**MTP2 候选仅5/8，门禁失败，未进行性能或长上下文测试。** 原始输出分别为加法`3848`（期望384）、减法`7636`（期望763）、提取`NPU7PU732329`（期望NPU7329），与前轮失败模式一致。A/B/A虽一致且无canary泄漏，不能覆盖三个错误答案。本轮未再观察到前轮的巨量申请，但计数与设备slot修复合用仍未恢复正确性，不能据此断言某个补丁单独有效或已定位驱动根因。

结论：统一公共参数和入口，保留经过验收的差异。a2继续MTP2/2048，clt继续MTP1/2048，RC继续MTP1/4096。三者共同C16/W4A8/Graph/GDN/MindIEQ/chunk512/max114688/max_num_seqs1。失败修复只保留在本实验记录，不进入runtime和默认启动代码，不为形式统一将a2降级。

实际视觉和prefix开关也必须明示：a2视觉加载/prefix off；clt language-model-only/prefix off；RC视觉加载/prefix on。视觉加载不代表图像业务验收。用户期望视觉和prefix默认启用，需要独立质量与内存验收，本轮为保证MTP复验条件一致未同时切换这两个开关。a2/clt利用率.95/.90、异步/同步调度不同，不能声称全部EP运行参数已经统一。

后续先收束可用服务/代码/制品来源，再独立对齐clt逐步draft/verify token、accept长度、slot/position和状态推进；其稳定错字模式值得定位，不能继续仅叠加RC修复。RC第二位置低接受是另一条实验线，保持独立，未证明与clt同根因。统一镜像工作应从固定源版本和已验收适配层出发，按架构/ABI构建，失败补丁不混入默认层。恢复验收以 `outcome.json`、`restored-mtp1-quality.json` 和 `state.txt` 为准。


恢复验收完成：`state=candidate-failed-restored`，health200，短题8/8、A/B/A一致且无canary泄漏。原容器继续18016服务，MTP2候选停止保留；运行容器另有原aisbench，不属于本次模型候选。恢复原始记录见 `restored-mtp1-quality.json`；汇总 `summary.json`，脚本/补丁哈希 `protocol-manifest.json`。
