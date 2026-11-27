# 310P ASR 交付制品与版本管理

2026-11-25。适用于 Confucius4-R2T2 与 Qwen3-ASR-1.7B 在单颗 310P3 上的交付。LLM 线（W8A8/TP2、W4/TP1）沿用 [固定来源的生产镜像构建](production-image-source.md) 与 [来源、修改与精度契约](provenance.md)，本文不改动那两条线。

## 与 LLM 线解耦

两条产品线的底座不同，不要为了"版本统一"把它们绑在一起：

| 项 | LLM 生产线 | ASR 建议基线 |
|---|---|---|
| 底座 | 生产恢复基底，合成 config，`llm-service-vllm-ascend@sha256:1a193d…` | 官方 `v0.30.0-310p`，锁 amd64 manifest |
| vLLM | `0.23.0+empty` @ `0fc695f` | `0.30.0+empty` @ `ced6857a` |
| vllm-ascend | `0.23.0rc2.dev93+g88eac2713` | `0.30.0` @ `c09d3e20` |
| CANN | `9.1.0-beta.1` | `9.2.1` |
| 额外补丁 | attention mask + triton 3.2.0 | 无（VAD reset 补丁在发布方 SDK 侧，不在 vLLM） |

**documentation verified**。vLLM-Ascend 文档已把 Qwen3-ASR 标为自 `v0.30.0` 停止维护。这改变了"是否追最新稳定版"的答案：ASR 侧**不追新**，升级不会带来该架构的适配改进，反而有被移除的风险；版本升级的价值只来自通用性能项，且必须先过下面的门槛。

## 交付制品的三元组

1. **底座镜像 digest**。锁 amd64 manifest，不锁多架构索引、不锁 tag。当前候选：
   - 索引 `sha256:041cc7f7b9e9c51cd823da28dc5a6c8ea1ac06df2b203e69a82766fc1668eafa`
   - amd64 manifest `sha256:ebea1420f0c31c3ef41344b64b8570326e78b9aee0ab3ba927646ce97131fcaf`
   - 取证见 [官方基础镜像替换评估](production-image-base-evaluation.md)。实验主体目前已在该 digest 上，这是选择它的主要理由。**2026-11-25 已确认采用该基线**，不跟 LLM 线共用底座。
2. **模型锁**。权重按 revision + 逐文件 SHA256 固定，不分发权重、不引用远端 main：
   - Confucius：`netease-youdao/Confucius4-R2T2`，revision `7eb29d6aa1779829512babab1825844b93ff43e2`，12 个文件见 [`configs/confucius4-r2t2-modelscope.lock.json`](../configs/confucius4-r2t2-modelscope.lock.json)。
   - Qwen3-ASR：沿用 voice-service-backends-py 的 `configs/models.lock.yaml` 约定（HF revision + 逐文件 SHA256）。
3. **SDK/启动覆盖层**。Confucius 的 `ws_server.py`（含 VAD reset 修补）与启动参数作为 overlay 打入派生镜像或挂载，不改底座内文件；overlay 自身记录 git commit 与文件 SHA256。

派生镜像只做"底座 + overlay"，不重编 vLLM 算子。需要打 vLLM 补丁时另立候选，按升级门槛验收，不进入默认路径。

## 镜像与 registry

- 命名：`registry.cn-hangzhou.aliyuncs.com/meetai/asr-310p`（与 voice-service 的 `REGISTRY`/`NAMESPACE` 一致）。
- tag：`<model>-<base>-<YYYYMMDD>-<gitshort>`，例如 `confucius4r2t2-v0300-310p-20261125-062726a`。
- **消费方只锁 digest**。tag 仅用于人读；每次推送后把 digest 回填到本仓锁文件与 deploy 配置，不回填就不算发布。
- 构建机与 310P 交付机分离，构建产物先在本机 `ychy-a2` 之外验证加载，再推 registry。
- `scripts/build-asr-310p-image.sh` 构建前校验：`Containerfile.asr-310p` 的 `FROM` 必须与锁文件 `base.image` 逐字一致，且该 digest 已在本地；不一致或缺失即退出。`--pull=never` 保证实际使用的底座就是这个 digest。

### 发布记录

| 项 | 值 |
|---|---|
| 镜像 | `registry.cn-hangzhou.aliyuncs.com/meetai/asr-310p:confucius4r2t2-v0300-310p-20261127-0aeb3c8` |
| manifest digest | `sha256:2da236236d47ca220fbd77a2762ebf343d65a6f3014b53960b05d7008a0d256d` |
| config digest | `sha256:50f54e0f51616fae7720584579befdb10b173d11a947542571227f0640e9ff16` |
| 构建提交 | `ascend310p-llm` `0aeb3c8` |
| 底座 | `quay.io/ascend/vllm-ascend@sha256:ebea1420f0c31c3ef41344b64b8570326e78b9aee0ab3ba927646ce97131fcaf`（amd64 manifest） |
| 推送机 | `ychy-a2`（有线；本机 n4 上行为共享公用带宽，不适合推 21 GB 产物） |
| 推送前验证 | `podman save` → n4 `podman load` → 容器内 import 通过；NPU 单场放音与归档终稿逐字一致（621 字，SHA256 相同），握手到首终稿 2052.313 ms 对归档 2057.048 ms |
| 锁文件 | [`deploy/image/asr-310p.lock.json`](../deploy/image/asr-310p.lock.json) 的 `publication` 段 |

消费方目前只有锁文件，尚无 Quadlet 或 deploy 配置引用该镜像；接入部署时按下节规则锁 digest。

## 升级门槛

版本升级（换底座、换 vllm-ascend、换 CANN、加补丁）必须同时满足：

1. 冻结语料的配对 CER/WER 不劣于当前交付 pin，逐段归一化终稿差异不超过基线自身变体量；
2. 两条参考负载（密集重叠、长会议）的首稳定文本 P95、EOS P95、PCM 排队 P95 不劣于当前 pin；
3. 稳定前缀/reset/增量帧序列与当前 pin 逐条一致；
4. 拒绝与异常断连计数不增加；
5. 记录新旧两套 digest 与向上/回滚步骤。

任一项缺失即不升级。当前 `v0.30.0-310p` 是**基线**，不是"最新"，也不需要被替换成更新的版本。

## 回滚

镜像不可变，回滚 = 把 registry tag 与部署配置指回上一份 digest 并重启；上一份 digest 保留在锁文件中至少一个大版本周期。模型权重不随镜像回滚，锁文件独立。

## 待定

- 同机部署：LLM 生产线 CANN `9.1.0-beta.1` 与 ASR 候选 `9.2.1` 能否共用一套 driver，需在交付机上验证；若不能，ASR 走独立机器或先做 CANN 升级评估。
- Qwen3-ASR 在 `v0.30.0` 的 `FULL_DECODE_ONLY` 与 batch 配置是否随底座升级变化，需在一次对照实验里确认。
- 派生镜像是否需要进 registry 的 public 命名空间、以及镜像层的复用策略，未定。
