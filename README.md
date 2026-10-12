# Ascend 310P — Qwen3.6 W4A8 / MTP / C8 实验

在Atlas 300I DUO / Ascend310P3上研究Qwen3.6-35B-A3B低比特推理。首版提供可运行的Routed Experts W4A8与BF16 MTP适配，以及独立C8分页KV算子实验。**W4A8+MTP+C8尚未联合运行。** 本项目不是华为、Qwen或vLLM官方发布。

| 能力 | 当前状态 |
| --- | --- |
| W4A8 TP1/TP2、FP16 KV | 已在a2启动、生成 |
| W4A8 + Qwen3.6自身BF16 MTP | TP1/TP2已运行，speculative tokens=1 |
| INT8 KV / C8 | 独立算子数值、动态页表、Graph与benchmark；未接完整模型 |
| 长上下文 | TP1无MTP精确128K生成已测；262144配置启动不等于实际256K验收 |
| 全面质量及稳定加速 | 尚未完成；INT4不保证更快 |

模型制品：[ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P)。发布状态及独立加载证据以[验证记录](docs/release-validation.md)为准。

## 精度与计算路径

仅routed experts量化为signed INT4，使用ModelSlim对称per-channel SSZ。scale按G128 kernel布局重复存储，**不是独立group-wise量化**。activation为设备端per-token INT8，使用INT4 limbs分解组合Cube整数计算。非专家张量保留原始浮点；MTP保留Qwen3.6自身19个BF16张量。text-only生成已测试，图像能力未验收。来源与社区贡献见[provenance](docs/provenance.md)和[NOTICE](NOTICE)。

## 固定环境与构建

已测镜像`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，CANN9.1，Torch2.10，torch-npu2.10.0.post4，SOC=Ascend310P3。SDK/driver由使用者安装，仓库不附带这些二进制。

在该镜像内挂载本仓库与一个可写工作目录，设置Torch backend autoload=0，禁用镜像不兼容Triton，source CANN环境后：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORK_DIR=/work/qwen36-build bash /project/scripts/build-native.sh
```

工作目录必须是新的隔离目录。脚本取固定社区commit，应用本仓库算子快照，构建3个OPP家族与独立`qwen36_int4` Torch namespace，不覆盖官方`_C_ascend`。构建与独立启动的实际验证范围见release-validation；首次第三方依赖下载受网络影响。

权重下载后先做完整性检查：

```bash
ms-hub download ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P --local-dir ./weights
python3 tools/validate_checkpoint.py ./weights --verify-hashes
```

## 启动服务

宿主使用rootless Podman；TP1只暴露davinci0，TP2额外暴露davinci1。这里的默认4K是首次复现门禁，长上下文应按实测内存预算单独设置。

```bash
MODEL_DIR=/absolute/path/weights \
OPP_DIR=/absolute/path/qwen36-build/artifacts/opp/vendors/custom_transformer \
EXTENSION=/absolute/path/qwen36-build/artifacts/qwen36_int4_custom.so \
TP=1 MTP=1 MAX_MODEL_LEN=4096 MEMORY_UTILIZATION=0.90 PORT=18015 \
bash scripts/run-service.sh
```

`MTP=0`关闭MTP；TP2设置`TP=2`。容器只挂载自包含模型目录，不依赖原始BF16或sidecar父目录。尚未支持C8开关；不要把`--kv-cache-dtype`直接加入此启动器声称已适配。

```bash
python3 tools/bench_generation_sanity.py --base-url http://127.0.0.1:18015 --model qwen36-w4a8 --out /tmp/generation.json --max-tokens 128
python3 tools/bench_http_stream.py --base-url http://127.0.0.1:18015 --model qwen36-w4a8 --prompt-tokens 128 --output-tokens 256 --concurrency 1 --repeats 3 --out /tmp/stream.jsonl
```

sanity只检查服务可用性、非空输出与明显异常，不替代准确率评测。benchmark计时含HTTP/调度；设备kernel性能单独采集profile。代码API以脚本`--help`为准。

## C8与技术反馈

[C8独立复现](docs/c8.md)包含选页NZ转换、原生PA对照、Cube/Split-KV候选和混合Mmad编译探针。需要真实310P执行；FP16工作区额外内存必须计入容量收益。

[华为反馈报告](docs/huawei-feedback.md)整理已知结果与五项具体问题：C8接口、混合Matmul、NZ/UB/L1流水、W4小M/Prefill与hybrid/MTP cache状态。下一节点是TP1短上下文真实C8生成，然后验证MTP回退和TP2。

## 从原始浮点重新量化

`tools/convert_modelslim_w4a8.py`与`modelslim_ssz_adapter.py`复用已安装ModelSlim SSZ，只输出expert sidecar；`prepare_hybrid_model.py`组合源模型与sidecar的实验目录，仍需导出自包含checkpoint。该组工具不会覆盖原始权重。ModelSlim版本应随重新转换单独锁定；当前公开制品的不可变标识为文件SHA256。

`python3 tools/export_checkpoint.py /path/to/hybrid /path/to/new-standalone`只复制index引用的tensor payload，逐tensor核对拷贝前后SHA256并生成新文件哈希；不重新量化、不保留源软链接。目标必须不存在。导出后运行`validate_checkpoint.py --verify-hashes`，再用隔离服务验证MTP和生成。
