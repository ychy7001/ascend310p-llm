# 固定来源的运行镜像

本入口把已提交的适配代码和已测制品打包，不需要挂载主机源码、扩展或OPP目录。权重和主机驱动继续独立挂载。它复用`configs/tp1-profile.sh`，与旧服务入口使用同一个参数解析器；不是重新实现量化后端，也不表示已从源码重建CANN或全部算子。

## 固定内容

- `deploy/image/base-amd64.json`固定registry digest、架构、vLLM/vLLM-Ascend源码SHA。构建时校验基础源码与工作树。
- `tools/prepare_runtime_image.py`通过`git archive`导出指定提交，忽略工作区未提交修改；仅选择运行必需文件，不包含benchmark、中间实验、模型权重或完整社区fork。
- 已测W4/C8/GDN/MindIEQ库与两个OPP树逐文件计算SHA256；host shared library检查ELF架构。CANN/MindIEQ许可证与本项目LICENSE/NOTICE一同封装。
- `/opt/qwen36/manifest.json`记录源码提交、基础镜像、制品来源路径和镜像内文件哈希。`verify-image.py`仅用标准库，不初始化NPU。
- clt原同步元数据适配收束为`QWEN36_SYNC_METADATA=1`显式开关；它是已验收MTP1的兼容条件，不是MTP2修复。所有失败MTP2补丁均未打包。

基础镜像重建依赖上游已发布digest仍可获取；算子二进制重建应另外按固定源码和SDK构建。本入口提供的是同一份已测制品的可重复打包，不宣称任意CANN/driver兼容或镜像逐字节可重现。

## 构建

在有Podman、对应基础镜像和已测制品的机器执行；`WORK_DIR`须为新目录。脚本从提交导出，因此先提交准备发布的实现，再构建。

```bash
source deploy/a2.env
export WORK_DIR=/home/ychy/ascend-lab-runtime/qwen36-image-NEW
python3 tools/prepare_runtime_image.py --revision HEAD \
  --mindieq-license /home/ychy/ascend-lab-runtime/mindieq-r3-20261013/ops-nn/LICENSE \
  --output "$WORK_DIR"
base=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["image"])' "$WORK_DIR/rootfs/opt/qwen36/manifest.json")
revision=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$WORK_DIR/rootfs/opt/qwen36/manifest.json")
podman build --pull=never --format docker --build-arg BASE_IMAGE="$base" \
  --label org.opencontainers.image.source=https://github.com/ychy7001/ascend310p-qwen36 \
  --label org.opencontainers.image.revision="$revision" \
  -t "localhost/qwen36:${revision}-amd64" "$WORK_DIR"
podman run --rm --entrypoint python3 "localhost/qwen36:${revision}-amd64" \
  -S /opt/qwen36/verify-image.py --base
```

该校验不需要设备映射。registry发布只在真实NPU门禁通过后使用唯一提交tag，并记录推送返回的digest；不覆盖现有stable。

## 启动

以a2的EP配置为例，先确保device0由此容器独占。`IMAGE`应设为已验收的registry digest，或本机候选tag；`MODEL_DIR`指向自包含权重。镜像自身提供全部适配代码和算子。

```bash
podman run --name qwen36-image-candidate --network host --group-add keep-groups --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v "$MODEL_DIR:/model:ro" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e PORT=18025 -e QWEN36_C8_ENABLED=0 -e NUM_SPECULATIVE_TOKENS=2 \
  -e MAX_MODEL_LEN=114688 -e MAX_BATCHED_TOKENS=2048 -e MEMORY_UTILIZATION=0.95 \
  -e ENABLE_PREFIX_CACHING=0 -e ASYNC_SCHEDULING=1 -e LANGUAGE_MODEL_ONLY=0 "$IMAGE"
```

默认C16、MTP2、Graph、GDN、MindIEQ E5、batch2048/chunk512、max114688、max_num_seqs1；prefix默认off，视觉加载，维持a2已测配置。prefix可显式设1，但产品默认开启的组合仍须独立验收。

clt同镜像候选需显式`NUM_SPECULATIVE_TOKENS=1 ASYNC_SCHEDULING=0 QWEN36_SYNC_METADATA=1 MEMORY_UTILIZATION=0.90 LANGUAGE_MODEL_ONLY=1`，其余公共参数不变。此描述是复现现有已验收条件；新镜像在clt的状态以results验收为准。MTP草稿数自动决定capture1,2或1,2,3，不需手工再改Graph参数。

ARM64 RC需要自己的固定基础镜像和ARM64 host扩展，不能运行amd64镜像。当前RC额外使用batch4096、同步调度、MTP1、prefix/视觉开启、blocking multimodal H2D及显式KV预算；这些差异不按hostname自动推断。尚未提供或验收新的ARM64镜像，不能照搬EP内存参数。

## 验收与回滚

新镜像需通过：镜像内哈希/基础源码 → 启动和算子解析 → 8题/A-B-A → 32K冷Prefill与Decode → 112K会议/长后质量。保存独立结果；失败不发布、不替换默认服务。候选只挂载模型和主机驱动，验收时应检查没有主机源码/算子bind。

停止systemd托管服务后才启动候选，避免同一芯片竞争。实验结束停止候选，再恢复原systemd服务并复验health/质量。旧容器若由`--rm`创建，停止后由原管理单元重建；保留原管理配置、源码和制品才能回滚。启动旧容器不会应用新的env。
