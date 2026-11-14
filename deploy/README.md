# deploy：三套部署档

每套档位一个子目录，装该档的启动脚本、环境变量示例与 systemd 单元。`image/` 是镜像构建上下文，`variants/` 是机器可读的档位登记。

档位之间互斥运行：同一张 310P 上不要同时起两套。切换必须停服、确认 NPU 空闲、冷却至少 60 秒再启动另一档（310P CANN TBE 初始化的既有观测）。

## 三套档位

| 档位 | 权重 / 并行 | 上下文 | MTP | 端口 | 状态 | 目录 |
| --- | --- | --- | ---: | ---: | --- | --- |
| W8A8 TP2 + native WY + MTP2 + 主模型 Graph | W8A8 / TP2 | 262144 | 2 | 18025 | **双卡默认交付**；eager mask 复用已过性能、精度与封装验收 | [w8a8-tp2-wy-mtp/](w8a8-tp2-wy-mtp/) |
| W8A8 TP2 + native WY（无 MTP） | W8A8 / TP2 | 114688 | 0 | 18025 | 已验证的 W8 基线，MTP 档的父镜像 | [w8a8-tp2-wy/](w8a8-tp2-wy/) |
| W4A8 TP1 + MTP + MindIEQ 长 Prefill | W4A8 / TP1 | 112K | 1 或 2 | 18024 | 稳定档，同时是 W8 的回滚镜像 | [w4a8-tp1/](w4a8-tp1/) |

默认版本、验收和 registry 维护见 [WY 默认交付](../docs/qwen36/w8a8-wy-delivery.md)；逐步复现见 [W8A8 262K 镜像复现](../docs/qwen36/w8a8-262k.md)。W4A8 档见 [Qwen3.6 文档](../docs/qwen36/README.md)。

## 各目录文件

### w8a8-tp2-wy-mtp/ — 双卡默认交付

| 文件 | 用途 |
| --- | --- |
| `start-wy-mtp.sh` | 镜像默认入口。全部参数走环境变量，含 capture size 推导与 MTP speculative-config 组装 |
| `qwen36-wy-mtp-image.env.example` | 直接调用镜像入口时的变量模板，固定 registry digest |
| `qwen36-wy-mtp-production.container` | a1 路径的 Quadlet 模板：两卡设备、60 秒冷却、失败重启 |
| `qwen36-wy-mtp-production.manifest.json` | 面向 a1 的完整部署描述，保留模型目录、18081 端口与服务名 |

固定引用：

```text
registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:75a20b2954edae93e5990f795e12c09c06faadcc49d6acd880cbc2f5f8ab217d
```

### w8a8-tp2-wy/ — W8 基线档

| 文件 | 用途 |
| --- | --- |
| `start-wy.sh` | 镜像默认入口，无 MTP 开关 |
| `qwen36-wy-image.env.example` | 变量模板 |
| `qwen36-wy-image.service` | systemd 单元 |

registry 引用：`...@sha256:e96595cbbcbabbde4fb39dd8b38ca882dcc3f13665af041ff456e2917c760b32`。

### w4a8-tp1/ — W4A8 稳定档

| 文件 | 用途 |
| --- | --- |
| `start.sh` | 镜像默认入口，读取镜像内 `configs/tp1-profile.sh` |
| `qwen36-image.env.example` | 变量模板 |
| `qwen36-image.service` | systemd 单元，`--pull=never`，只跑本机已预加载镜像 |

profile 选择（`QWEN36_PROFILE`，默认 `long-prefill`）与参数矩阵见 [Profile 说明](../docs/qwen36/profiles-20261014.md)。

### image/ — 镜像构建上下文

| 文件 | 用途 |
| --- | --- |
| `Containerfile` | 通用构建模板：`COPY rootfs/ /` 后跑 `verify-image.py --base` |
| `verify-image.py` | 在镜像内校验 manifest 与文件 SHA256，不导入 torch、不初始化 NPU |
| `base-amd64.json` | W4A8 档的基础镜像锁（quay.io digest + vLLM/vLLM-Ascend 源码 SHA） |
| `base-wy-310p.json` | W8A8 档的基础镜像锁，staging 专用，不与 W4A8 的锁互相推断 |

`image/` 里**不放**启动脚本。启动脚本属于各档位子目录，构建时由 `tools/prepare_*image.py` 复制进上下文。

### variants/qwen36.json

三档的机器可读登记：入口脚本、配置、基础镜像锁、registry digest、已验收范围。改档位时同步更新，不要只改 `.md`。

## 镜像内路径与仓库路径

镜像内的 `/opt/qwen36/start.sh`、`/opt/qwen36/start-wy.sh`、`/opt/qwen36/start-wy-mtp.sh` 是**已发布镜像身份的一部分**（已发布 digest 的 `Entrypoint` 就是这些路径），仓库整理时不得改动。仓库内路径可以调整，对应关系是：

| 仓库内 | 镜像内 |
| --- | --- |
| `w4a8-tp1/start.sh` | `/opt/qwen36/start.sh` |
| `w8a8-tp2-wy/start-wy.sh` | `/opt/qwen36/start-wy.sh` |
| `w8a8-tp2-wy-mtp/start-wy-mtp.sh` | `/opt/qwen36/start-wy-mtp.sh` |

三个路径被 `tools/prepare_runtime_image.py`、`tools/prepare_wy_runtime_image.py`、`tools/prepare_wy_mtp_image.py` 硬编码引用。移动档位目录时必须同步改这三个工具和 `variants/qwen36.json`，否则镜像构建上下文会缺文件。

## 镜像构建

```
tools/prepare_runtime_image.py      → image/Containerfile → w4a8-tp1/start.sh
tools/prepare_wy_runtime_image.py   → W8 基线镜像
tools/prepare_wy_mtp_image.py       → 在 W8 基线镜像上叠加兼容层
```

三个工具都只打包已有的、已测过的二进制与源码快照，不重新编译 CANN 或算子。构建与已验收范围详见 [runtime-image](../docs/qwen36/runtime-image.md) 与 [runtime-lineage](../docs/qwen36/runtime-lineage.md)。

## 回滚

默认 W8A8 先回退同档上一版 WY digest，模型、TP2/MTP2 及上下文配置可以保留。仅需关闭复用时设 `QWEN36_EAGER_MASK_REUSE=0` 后重启。单卡降级才选 W4A8 TP1。完整引用见 [WY 默认交付](../docs/qwen36/w8a8-wy-delivery.md)。换镜像步骤：

1. 停当前容器并移除。
2. 确认 NPU 空闲、两服务均为 inactive。
3. 冷却至少 60 秒。
4. 启动目标档，要求 `/health` 加一次正常生成。
5. 回滚同样执行完整冷却，不做即时镜像互换。

不能只换 manifest 而继续用旧镜像，也不能把旧的 `Exec` 直接附加到新镜像的 `ENTRYPOINT` 后面。
