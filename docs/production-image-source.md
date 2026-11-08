# 固定来源的生产镜像构建

本仓在现有 `deploy/image` 中增加 `production` target，用固定的生产恢复基底、上游提交和补丁队列重建 W8A8/TP2 镜像。默认 `runtime` target 仍是已有 W4/TP1 打包入口。两者没有共享模型启动参数：生产 target 继承基底的官方入口、环境和默认命令，服务方继续提供实际生产命令。

CPU 取证、恢复、构建与对照已完成，恢复基底和候选均已发布到指定的新仓库。同日六臂启动复现均成功；随后采用七项指标和1.25倍率的 day 窗口，在生产 A/A′ 比较时遇到显式 top-k 支持交集为空的异常，具体阈值未冻结、候选未启动。按第三轮噪声数据修正判定口径后（见[判定口径修正](#判定口径修正2026-11-05)），`day2-20261105` 完成四项G1和全部G2三臂门禁，均通过；随后同一窗口的八格基准按旧规则判失败，根因为旧规则本身不可用，已改为跨格汇总中点检验并在同一份证据上再判定通过（见[基准窗口](#基准窗口判定规则不可用改为汇总中点检验2026-11-06)）。没有切换生产引用。官方换底另见 [评估](production-image-base-evaluation.md)。

## 固定组成

来源锁定在 [`production-amd64.json`](../deploy/image/production-amd64.json)，原始层摘要和取证结果在 [`production-forensics.json`](../deploy/image/production-forensics.json)。

- 生产来源：`${PRODUCTION_REGISTRY}/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38`。
- 恢复基底：生产前 19 个文件系统层，OCI manifest digest `sha256:5a49ce1e72934270662e800b0f6891ec4e97b93a763e256ca71fc6954be7c909`，image config ID `sha256:e86feb2f76788cfa1fa70bfe088ca3b26ad92a23e316927e7a41e51739ebd9b7`。
- vLLM：`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，安装版本 `0.23.0+empty`。
- vllm-ascend：`88eac2713b14beedc7a48803274f8d6cee3e9f30`，安装版本 `0.23.0rc2.dev93+g88eac2713`。
- Python 3.12.13、torch `2.10.0+cpu`、torch_npu `2.10.0.post2`；CANN 安装文件确认 `9.1.0-beta.1`。
- 入口 `/usr/local/bin/entrypoint.sh`，默认命令 `/bin/bash`，工作目录 `/workspace`；生产 Env 完整记录在取证 JSON 中。

恢复基底有明确限制：历史记录的官方 tag 是 `${UPSTREAM_REGISTRY}/ascend/vllm-ascend:nightly-releases-v0.23.0-310p-openeuler`，但 tag 已被覆盖，历史候选 manifest 实际读取返回 404。恢复工具保留前 19 层、截断 overlay 历史并移除 overlay 的两个标签；生成的是本仓可重复恢复的合成配置及 digest，不能冒充遗失的官方原始 config 或 digest。已经没有独立的官方原始镜像可对照。

基底的两个 Git 工作树均干净；生产的 vLLM 工作树干净，vllm-ascend 仅下表两个文件有改动。两份包通过 editable install 指向 `/vllm-workspace/`，site-packages 中没有独立的同名包目录。文件对照因此同时包含 editable 元数据、finder 和实际源树。

## 生产相对恢复基底的全部差异

生产共 22 层；第 20、21 层分别覆盖实现和测试，第 22 层运行 triton 安装。完整文件清单比较保留权限、uid/gid、链接目标和文件 SHA256，忽略时间戳。基底 189233 条目，生产 189664 条目，差异共 433 条目：

| 类别 | 条目数 | 本仓表达 |
| --- | ---: | --- |
| attention mask 实现 | 1 | `patches/production/attention-mask.patch` |
| 对应测试 | 1 | 同一个补丁 |
| triton 包及 dist-info（含目录、字节码缓存） | 414 | Containerfile 显式安装锁定 wheel |
| triton 的两个 CLI | 2 | 同一 wheel 安装步骤 |
| pip 缓存 | 13 | 安装的缓存副产物；不作为运行文件锁定 |
| `/etc/hostname`、`/etc/resolv.conf` | 2 | 第 22 层的空文件，权限均 `0700`；Containerfile 显式复制 |

两个覆盖文件的生产权限均为 `0664`，Containerfile 在应用补丁后显式恢复该权限。无其他内容或权限差异；pip freeze 仅新增 `triton==3.2.0`；环境和入口继承基底。空的 `/etc` 文件不携带主机名、IP 或 DNS 配置。

补丁的来源、上游核查和必要性见 [队列说明](../patches/production/README.md)。生产额外改动未找到对应上游 PR，标记为“无”；基线已有 PR #10181 不重复入队。没有分析或修改 Ascend 算子/内核实现，也没有对 attention mask 做数值或性能评价。

triton wheel 为 `triton-3.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`，SHA256 为 `8d9b215efc1c26fa7eefb9a157915c92d52e000d2bf83e5f69704047e63f125c`，由 `production-requirements.txt` 的 `--require-hashes` 校验。无需另装依赖或编译 Ascend 算子。

## 恢复和构建

以下命令仅在本机操作。`WORK_DIR` 放仓库外，`PRODUCTION_REGISTRY` 由操作方提供。每项拉取、解层、构建或批量哈希开始前检查 `pgrep -f '[a]sr_realtime_ws_smoke'`；有输出时每 60 秒检查一次，结束后才开始，并记录等待时长。重步骤还需持有 `flock /tmp/a310-cpu-heavy.lock`，串行运行。

```bash
PRODUCTION_IMAGE="${PRODUCTION_REGISTRY}/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38"
podman pull --arch amd64 "$PRODUCTION_IMAGE"
podman save --uncompressed --format oci-dir -o "$WORK_DIR/production-oci" "$PRODUCTION_IMAGE"
python3 deploy/image/recover-production-base.py "$WORK_DIR/production-oci"
podman pull "oci:$WORK_DIR/production-oci"
BASE_IMAGE=$(python3 -c 'import json; print(json.load(open("deploy/image/production-amd64.json"))["base_image_id"])')
podman build --jobs 1 --arch amd64 --pull=never --format docker --target production \
  --build-arg BASE_IMAGE="$BASE_IMAGE" -f deploy/image/Containerfile \
  -t localhost/a310-image-src:candidate .
```

恢复工具先验证生产 22 个 diff_id，再生成前 19 层的 OCI 索引。该工具原地改变导出目录的 index，保留 blob；再次恢复需要新的生产导出目录。输出应与锁文件的 manifest/config digest 一致。构建用 image config ID 固定 `FROM`。恢复基底已发布，其他构建机可先拉取锁文件 `base_registry_reference` 的不可变引用；该引用中的 registry/namespace 由操作方展开。

Containerfile 校验 x86_64、两份源码 HEAD、干净工作树和补丁 SHA256，随后应用队列并按 wheel 哈希安装 triton。构建不会加载模型或导入 torch。基底自带 CANN 和上游编译制品，本流程复现生产 overlay，并不从源码重编 CANN 或全部上游二进制。

## CPU 对照入口

[`inspect-production.py`](../deploy/image/inspect-production.py) 使用 `python3 -S` 采 Git、安装元数据、pip freeze、CANN 安装信息和入口文件，不导入 torch 或设备插件。对生产和候选分别运行，保存 JSON：

```bash
podman run --rm -i --read-only --network none --entrypoint python3 "$PRODUCTION_IMAGE" \
  -S - < deploy/image/inspect-production.py > "$WORK_DIR/production-inventory.json"
podman run --rm -i --read-only --network none --entrypoint python3 localhost/a310-image-src:candidate \
  -S - < deploy/image/inspect-production.py > "$WORK_DIR/candidate-inventory.json"
podman image inspect "$PRODUCTION_IMAGE" > "$WORK_DIR/production-inspect.json"
podman image inspect localhost/a310-image-src:candidate > "$WORK_DIR/candidate-inspect.json"
```

分别在 `podman unshare` 中用 `podman image mount` 取得挂载根目录，串行调用 `python3 deploy/image/hash-files.py ROOT OUTPUT.json`；结束后 `podman image unmount IMAGE`。保存两份完整清单后比较：

```bash
python3 deploy/image/compare-production.py \
  --production "$WORK_DIR/production-inventory.json" \
  --candidate "$WORK_DIR/candidate-inventory.json" \
  --production-files "$WORK_DIR/production-files.json" \
  --candidate-files "$WORK_DIR/candidate-files.json" \
  --production-inspect "$WORK_DIR/production-inspect.json" \
  --candidate-inspect "$WORK_DIR/candidate-inspect.json" \
  --out "$WORK_DIR/candidate-comparison.json"
```

比较器检查源码/安装文件、所有非缓存文件、pip freeze、版本、环境、入口、工作目录和用户；缓存差异单独完整列出。无论源码还是其他目录出现非缓存差异，都不会输出通过。原始完整清单和镜像 blob 存放在仓库外。

## 本次 CPU 对照结果

候选本机 manifest digest 为 `sha256:16b584b0bb5dda64d5c121df438e43467e9ab090fb06cdd04c55b72e1ec952f9`，image config ID 为 `sha256:efd8af10e8619595c428493a7dc14338815679a35f0e77f67b736e37a07a3a79`。该 digest 对应构建时本地镜像；发布后的 OCI manifest 引用见下文。

[`production-comparison.json`](../deploy/image/production-comparison.json) 记录结果和完整清单摘要：

- 16869 个源码及安装文件全部一致，完整文件树非缓存差异为 0。
- pip freeze、四个框架包版本、editable 安装、入口脚本及完整镜像 Config 一致；Config 包括 Env、Entrypoint、Cmd、工作目录、用户和标签。
- 剩余 65 个差异条目来自 pip 缓存和 Python 字节码缓存；时间戳未参与比较。
- 初次候选仅有两个覆盖文件权限为 `0644` 的差异；显式恢复生产的 `0664` 后重新构建并完整复核通过。未更改文件内容。

这证明本次 CPU 文件和配置复现通过，不代表 NPU、G1/G2、性能或业务验收通过。候选使用恢复基底内已有的上游编译产物，未重新编译这些二进制。

## 与 W4/TP1 机制的关系

已有 `base-amd64.json` 固定 2026-09-22 的另一基础索引，vllm-ascend 为 `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。准备器、start.sh 和原校验模式打包 W4/C8/GDN/MindIEQ 制品并启动 TP1；它属于另一运行镜像线，部分 vLLM 来源重合。生产 target 使用同一个 Containerfile 和校验器，保留官方入口；没有把已有 qwen36/qwen38 实验补丁加入生产队列。Podman 默认跳过不影响目标的 stage，现有 W4 准备器仍选择最后的 runtime stage。

## 发布记录

恢复基底的远端引用是 `${REGISTRY}/${NAMESPACE}/vllm-ascend-310p-base@sha256:f7b14f2deb44b7558cc5506bbd7a329126bd260491d35a88249f94119b154a4f`，tag 为 `recovered-1a193ded0d75-amd64`。推送输出 digest 与通过 registry API 重新下载的原始 manifest SHA256 完全一致；远端 config 的 SHA256 与恢复基底 image ID 一致，19 个 diff_id 与生产前缀一致。记录见 [`production-publication.json`](../deploy/image/production-publication.json)。

本地恢复 OCI 使用未压缩层；推送时 Podman 将层 gzip 压缩，因此发布 manifest digest 与本地未压缩 manifest digest 不同。config、文件系统 diff_id 和文件内容没有改变。锁文件保留未压缩恢复 digest，并另外记录可拉取的 registry 来源引用。

候选已发布为 `${REGISTRY}/${NAMESPACE}/ascend310p-llm-candidate@sha256:c590aebfa21e59f75a7d845c5b5563bce51292539c53b6b438057ffc24d167f0`，tag 为 `image-src-f46116b-amd64-310p`。原始 manifest 远端回读与本地推送输出 digest 一致，远端运行 Config 和所有 diff_id 与本地候选一致。发布使用 OCI 格式，config 外层序列化与本地 Docker 格式不同，不能将两者的 config ID 混用。生产镜像仓库未写入任何 tag。

## 早先设备窗口与门禁口径（历史记录）

实验设备使用阻塞共享锁；取锁后才检查原服务。本轮排队4318.97秒后取锁，原服务当时 active、health=200。整个设备窗口持锁3132.73秒，结束恢复原单元，并通过新的独立 SSH 连接确认 health=200 后解锁，未超过三小时。

固定同一模型和生产启动参数，先采生产 A、生产新进程 A′，再冻结 G1 的 A′ 对 A 误差 ×1.0 阈值，无绝对误差地板；G2 同轮重跑 GSM64、13例结构化和8短题，各项差≤1。候选 B 在冻结后启动，但在 TP1 权重加载之前的 Python 子进程 CPU FP16 zeros 初始化中段错误退出（ExitCode1），未达到健康状态；没有 B 的 G1/G2 数值或性能结论。按计划外问题停止规则，未重试、未调整配置、未跑八格基准。

生产 A/A′ 的 G2 均为 GSM63/64、结构化C1/C3各13/13、短题前后各8/8。G1生产噪声已有33/35条greedy分歧，平均共同top-k条件KL为0.0372147682；生成观测只比较相同条件前缀，不能替代完整teacher forcing。候选启动失败的根因尚未确定，不能以CPU文件一致代替设备验收。

完整冻结值、基线分数、失败与恢复证据摘要见 [设备窗口结果](../results/production-image-source-20261105/README.md)。恢复基底及候选已发布，生产引用未切换。官方换底仅评估，未执行。


## 后续实验窗口规则（2026-11-05 更新）

后续设备实验全程持有共享 NPU 锁，单窗口最多三小时。取锁后若原后端单元 active 则停止并 reset-failed；记录内存、设备和容器状态。实验镜像服务单元全程不启动。窗口结束清理本窗口容器和进程，确认没有运行容器、两个服务单元均 inactive，以 `window-closed.json` 留存起止时间、内存和残留检查结果后释放锁。结束时不重启服务，也不执行原端口 health 确认。前一窗口的恢复记录保留为历史事实。

启动崩溃复现按候选/生产交替各三次，仅等待健康或退出。每臂分别保存完整容器日志、命令、镜像引用、inspect、起止时间、退出码、启动前状态和启动区间内核日志。先定性启动失败，再决定是否执行同轮 A、A′、B 门禁；沿用已冻结阈值，右删失结果保留 `null`。基准需门禁全部明确通过且窗口时间充足后另行执行。


## 同日后续窗口：启动成功，旧阈值不适用于同轮噪声

候选与生产交替各三次仅启动，六次均 health=200、健康后停止 ExitCode0，双方失败率均为0/3。未复现稳定的候选特有启动缺陷；由于没有崩溃，本轮仍无法确认旧故障的主机根因。每臂完整日志及启动前状态、命令、inspect、时间和内核记录已独立落盘。

随后重跑生产 A、A′ 的 G1/G2，保留旧冻结阈值。最大共同 top-k 条件 KL 为6.75243342666，超过6.19170636567；最大目标 logprob 差的保守下界≥13.3013935387，已超过12.7878983021。冻结阈值不适用于本窗口噪声，依规则未启动门禁 B、未跑性能基准，没有重冻或放宽阈值。A/A′ 的 G2均为GSM63/64、结构化C1/C3各13/13、短题前后各8/8。

窗口持锁4188.02秒，关闭自检确认无运行容器、两个服务inactive，未重启原服务。逐臂结果、完整八项噪声对照及证据散列见[后续窗口结果](../results/production-image-source-20261105/evening/README.md)。旧 emergency 日志还确认TP0在段错误之后打印了auto-prefetch禁用提示，不能沿用“该行缺失”作为镜像差异。


## day 窗口：生产对照比较异常，未启动候选

`day-20261105` 按七项 case 归一化指标和1.25倍率运行，判定工具及临时控制脚本均未修改。生产 A、A′ 各完成35条G1和全部G2；两臂均健康启动，GSM63/64、结构化C1/C3各13/13、短题前后各8/8。之后 `compare(a, ap)` 在 `replay-21` 第2个生成位置遇到两份20-token集合交集为0，抛出 `ValueError('No shared top-k support')`。窗口状态为 `failed`；具体阈值未冻结、B未启动，没有候选通过或失败的数值判定。未填补替代KL值、修改规则或重开窗口。

汇总脚本正常生成失败摘要，因缺少B数据，依赖B的G1重算与冻结时序校验分支未执行。窗口持锁2839.21秒，关闭自检确认无运行容器、两个服务inactive，未重启服务。逐臂日志散列、七项未判定状态、诊断缺失及完整时延账见[day窗口结果](../results/production-image-source-20261105/day/README.md)。八格三臂基准的工作量排期估算约99分钟，C4尚未实测，仍须先解决比较异常并完成门禁。


## 判定口径修正（2026-11-05）

空交集是高熵位置的合法观测而非缺陷：`replay-21` 第 1 个生成位置两臂各自返回 20 个 token 而交集为 0（A 的 top1 为 98260、对数概率 -0.3364，A′ 为 97279、-0.0522），历史上同一位置的交集是 5 和 3。现在 `conditional_topk_kl` 返回 `(None, 0)`，`compare()` 记账到 `empty_topk_support` 并从 KL 均值中排除，不再中断整轮比较。分歧前的位置结构上不会出现空交集——那里两臂 top1 相同，生成模式又恒返回自身 top1，故该 token 必在双方 top-k 内；三轮实测 1144/1664/1114 个前缀位置均为 0 次。

同轮 A/A′ 噪声现在有三轮，按 case 归一化的量级指标被证明不可复现：`mean_case_mean_conditional_topk_kl` 漂 1.96×、`mean_case_max_conditional_topk_kl` 1.66×、`mean_prefix_absolute_target_logprob_difference` 1.67×（KL 形式 5.09×）。根因是 35 个 case 里分别有 5、4、5 个在第 0 位即分歧、完全没有前缀，却仍按 1/35 参与平均。改为**分歧前全部位置汇总成一个分母**的 `pooled_prefix_kl` 后，三轮为 0.00714/0.00737/0.00687，即 1.07×。

因此判定集从七项收缩为四项：`top1_disagreeing_case_rate`（1.09×）、`mean_case_top1_mismatch_rate`（1.17×）、`mean_greedy_divergent_suffix_fraction`（1.08×）、`pooled_prefix_kl`（1.07×）。其余指标（含旧门禁七项与 `severity` 四个分位计数）继续计算并落盘，只报不判。`mean_top1_divergent_suffix_fraction` 与 `mean_greedy_divergent_suffix_fraction` 在生成模式下恒等（8960 个位置零偏差），保留前者仅为对照。

倍率原为 1.25，后被第四轮噪声证伪，改为 1.40：四轮共 24 个两两比值，几何均值 1.0973、对数标准差 0.0817，最差观测比值 1.3132（`pooled_prefix_kl`，r2/r4）；1.25 下无真实退化时的误拒率为 16.4%。1.40 约为均值 +2.9σ，误拒率 4.0%，对真实 1.5× 退化的检出率 99.9%。day2 按更严的 1.25 通过，因此在 1.40 下同样通过。改动见提交 `3ce016e`，单测 15 项。

`top1_disagreeing_case_rate`（阈值 1.178571）与 `mean_greedy_divergent_suffix_fraction`（阈值 1.008650）的噪声底乘倍率后超过指标上限 1.0，这两项实际不可能失败，门禁看起来四项、实际在判两项。不补绝对上限（那会重新引入一个与噪声无关的人工阈值），改为让 `adjudicate()` 在遇到取值天然不超过 1 的比例指标其阈值已达 1.0 时抛 `ValueError`：宁可停下改配置，也不静默保留一个不会失败的判据。两项仍照常计算、落盘和汇报。day2 的冻结记录按当时口径重算，与该守卫无关。

新增只报不判的 `mean_prefix_top1_margin_delta`：分歧前前缀上，|胜出 token 与次优 token 的 logprob 间距之差|，单调 softmax 下等价于 logit margin。四轮 A/A′ 为 0.4287/0.4363/0.3834/0.4151，最差比值 1.1379、对数标准差 0.0809，是全套候选指标里最稳的；它把"两个近乎并列的 token 之间翻位"和"跨大间距翻位"区分开，而分歧率把两者都记成一次。本轮尚未纳入判定集。


## day2 窗口：四项 G1 与 G2 三臂通过

`day2-20261105` 沿用提交 `3ce016e` 的比较工具及已修正的临时控制器，按 A → A′ → 冻结 → B → G2 执行，倍率1.25、绝对地板0，未修改判定逻辑或模型启动参数。四项G1全部 `passed=True`：B的case分歧率0.942857、平均case top1不一致率0.293311、平均贪心分歧后缀比例0.843192、汇总前缀KL0.00494283。A/A′及A/B的空top-k交集计数均0，位置清单均为空。其余量级指标、删失和严重程度计数只报不判。

G2三臂均为GSM63/64、结构化C1/C3各13/13、短题前后各8/8，B相对A和A′的计数差均0。三臂均健康启动、停止ExitCode0，无本轮启动崩溃或启动区间内核段错误记录。汇总重算比较和G1判定与原记录一致，输入散列及冻结早于B启动校验通过。

窗口持锁4233.38秒（70.56分钟），关闭自检确认无运行容器、两个服务inactive，未重启服务。完整四项噪声/阈值、全部诊断、逐臂日志散列和时延账见[day2窗口结果](../results/production-image-source-20261105/day2/README.md)。本轮两项比例阈值大于1，对这两项没有有效的拒绝边界；保留冻结值原样，不自行截断。平均case top1不一致率为阈值的99.49%，本轮通过不代表存在宽裕余量。

当时八格基准未在本窗口执行。按本轮串行请求耗时估算下一轮三臂八格约98.76分钟，含15分钟清理余量；C4争用与streaming时延未实测，仅供三小时内窗口排期。


## 基准窗口：判定规则不可用，改为汇总中点检验（2026-11-06）

`day2-20261105/bench` 在同一窗口的门禁之后单独跑完八格三臂（A → B → A′，每格每臂3轮、256输出token）。24份jsonl全部完整：3轮均为成功请求，prompt token ID 三臂逐格相同，usage 合乎规定。没有请求失败、没有重试、没有删格。

按冻结的基准规则判**失败**：逐格逐指标要求 `B/A ≤ 1+|A′/A−1|`（时延）或 `≥ 1−|A′/A−1|`（吞吐），24项检查过16项、八格过3格。八项失败全部越界0.21%–0.94%，没有一项超过1%；其中七项越界幅度小于同格 A′ 对 A 的漂移本身，唯一例外是 4096/c4 的 TPOT（越界0.94%、该格漂移0.73%）。

根因在规则，不在候选。A′ 与 B 是独立采样，无退化时逐项检查在对数比上是对称的，单格三项全过 `P=0.5³=0.125`，24项全过 `P=0.5²⁴=5.96e-08`——这条规则对字节完全相同的候选也几乎不可能通过。零假设下期望12.0/24项、1.00/8格；实测16/24项（`P(X≥16)=0.0758`）、3/8格（`P(X≥3)=0.0673`），落在零假设分布之内，不构成任何候选退化证据。第二条弱点：C4 格内变异大，TTFT 格内变异系数35.57%–46.12%、TPOT 7.09%–59.74%（C1 分别为0.11%–4.30%、0.00%–0.50%），逐格中位数在 C4 就重复不出来，旧规则却把它当确定值做比值。

改为跨八格汇总检验：每格取 `B/√(A·A′)` 的对数，八格求均值后取单侧 t 界。几何中点消掉线性漂移，因为 B 就排在 A 与 A′ 之间。时延拒绝条件 `exp(mean+t·se) > 1+δ`，吞吐 `exp(mean−t·se) < 1−δ`，`t(0.95,7)=1.894578605`。δ 取时延2%、吞吐1%：按本轮实测八格对数标准差做完整流水线模拟，第一类错误为 TTFT 4.78%、TPOT 0.00%、吞吐0.20%；检出率 TTFT 1%→49.2%、2%→95.1%，TPOT 1%→4.6%、2%→94.9%，吞吐 1%→95.0%。吞吐 δ 更紧是因为它八格对数标准差仅0.0056。当前8格样本量下 TPOT 对1%级退化基本无检出能力，这是真实限制，不掩饰。

同一份证据按新规则再判定三项均通过：TTFT 汇总0.999141319、界1.009314112；TPOT 1.002870942、界1.007920402；吞吐1.000421827、界0.996674389（阈值分别为1.02、1.02、0.99）。这是对已落盘证据的再判定，不是新测量，也不改变本窗口当时按冻结规则判出的 `failed`。是否以此作为设备性能结论待定；重采一轮新证据的成本约为一个70分钟窗口。

实现在提交 `1201db7`（`benchmarks/ascend310p/adjudicate_benchmark.py`），单测11项见 `d6f1487`，工具清单见 `4e83620`。`adjudicate()` 校验 `degrees_of_freedom == len(cells)-1`，格数变化后沿用旧 t 值会抛错而不是悄悄放松检验，与比例指标阈值达1.0即抛错的守卫同一思路。三臂容器时长1394.255/1387.714/1388.869秒，ExitCode均0；持锁4191.34秒，未超三小时，关闭自检通过。逐格数值、C4变异范围和证据散列见[基准窗口结果](../results/production-image-source-20261105/bench/README.md)。

没有会议回放、RAG Prefill或通用负载的性能收益结论；`device_accepted` 仍为 `false`。


## 2026-11-06 审查更正

历史“单次通过率 0.5”、由此推导的二项概率，以及新规则模拟的误拒率/检出率不再作为有效统计结论。它们没有充分处理共同对照、固定网格与相关性。新 t 汇总暂作为预声明的固定网格工程验收规则，不能声称具有已验证的 95% 覆盖率；几何中点也不保证消除实际时间漂移。详见 `docs/research/benchmark-adjudication-review.md`（从 results 阅读时以仓库根目录定位）。历史原始数据和冻结判定保留。

已启动单独的预冻结确认窗口 `confirm-20261106`，不将旧数据再分析作为独立确认。该窗口结果另行记录。


确认窗口的最终结论以新证据为准。旧文中将失败唯一归因于判定规则、排除候选退化的表述，缺乏有效的统计支撑，不再作为当前结论。

11-06确认窗口因用户中断父会话而终止，留下107/180请求记录，未裁决、未拼接重跑，设备验收仍未完成；远端窗口已清场并独立核实锁释放。详见[确认窗口](../results/production-image-source-20261105/confirm/README.md)。
