# 固定来源的生产镜像构建

本仓在现有 `deploy/image` 中增加 `production` target，用固定的生产恢复基底、上游提交和补丁队列重建 W8A8/TP2 镜像。默认 `runtime` target 仍是已有 W4/TP1 打包入口。两者没有共享模型启动参数：生产 target 继承基底的官方入口、环境和默认命令，服务方继续提供实际生产命令。

本次只做 CPU 取证、恢复、构建与对照。没有执行模型推理、设备门禁或切换生产引用。官方换底另见 [评估](production-image-base-evaluation.md)。

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

以下命令仅在本机操作。`WORK_DIR` 放仓库外，`PRODUCTION_REGISTRY` 由操作方提供。每项拉取、解层、构建或批量哈希开始前检查 `pgrep -f '[a]sr_realtime_ws_smoke'`；有输出时每 60 秒检查一次，结束后才开始，并记录等待时长。重步骤串行运行。

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

恢复工具先验证生产 22 个 diff_id，再生成前 19 层的 OCI 索引。该工具原地改变导出目录的 index，保留 blob；再次恢复需要新的生产导出目录。输出应与锁文件的 manifest/config digest 一致。构建用 image config ID 固定 `FROM`；基底没有发布到 registry，不提供虚构的 registry 引用。

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

候选本机 manifest digest 为 `sha256:16b584b0bb5dda64d5c121df438e43467e9ab090fb06cdd04c55b72e1ec952f9`，image config ID 为 `sha256:efd8af10e8619595c428493a7dc14338815679a35f0e77f67b736e37a07a3a79`。未推送，该 digest 仅对应本地镜像，不能作为已发布 registry 引用。

[`production-comparison.json`](../deploy/image/production-comparison.json) 记录结果和完整清单摘要：

- 16869 个源码及安装文件全部一致，完整文件树非缓存差异为 0。
- pip freeze、四个框架包版本、editable 安装、入口脚本及完整镜像 Config 一致；Config 包括 Env、Entrypoint、Cmd、工作目录、用户和标签。
- 剩余 65 个差异条目来自 pip 缓存和 Python 字节码缓存；时间戳未参与比较。
- 初次候选仅有两个覆盖文件权限为 `0644` 的差异；显式恢复生产的 `0664` 后重新构建并完整复核通过。未更改文件内容。

这证明本次 CPU 文件和配置复现通过，不代表 NPU、G1/G2、性能或业务验收通过。候选使用恢复基底内已有的上游编译产物，未重新编译这些二进制。

## 与 W4/TP1 机制的关系

已有 `base-amd64.json` 固定 2026-09-22 的另一基础索引，vllm-ascend 为 `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。准备器、start.sh 和原校验模式打包 W4/C8/GDN/MindIEQ 制品并启动 TP1；它属于另一运行镜像线，部分 vLLM 来源重合。生产 target 使用同一个 Containerfile 和校验器，保留官方入口；没有把已有 qwen36/qwen38 实验补丁加入生产队列。Podman 默认跳过不影响目标的 stage，现有 W4 准备器仍选择最后的 runtime stage。

## 设备验证的交接边界

第 1–3 步结论须交 owner 确认，第 5 步另行派发。本轮未登录交付站、未停服务、未运行 NPU 门禁。

后续若获批，在无业务流量的限时窗口内停止 backend，固定同一模型及有效参数，以生产 A、候选 B、生产重启 A′ 三臂执行现有 G1/G2 工具；先用 A/A′ 冻结噪声阈值，再采 B。沿用 `tools/compare_token_logprobs.py`、GSM64、短题/复用/事实和业务检查；不以自动 schema 通过代替语义验收。门禁通过后运行 `tools/bench_http_stream.py` 的 1K/4K/16K/32K × C1/C4 八格矩阵，不要求额外收益，候选不得比生产差。窗口结束恢复原镜像，确认 `18081/health=200` 且 `backend-api-smoke` 通过。具体时长、冻结样本和阈值由后续派发确定。
