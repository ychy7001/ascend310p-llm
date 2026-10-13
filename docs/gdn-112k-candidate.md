# TP1 / 48G / GDN 优化候选

2026-10-13，a2 的单颗 Ascend 310P3。基于固定 GDN PR #17640 head `a2d228c7850f1749a4a934cc7514cd6c01fd7948`，保留 N128 W4 算子和动态 C8。该候选是本次已测参数的工程选择，不声称全局最优或正式产品验收。

## 参数选择

| 配置 | 候选值 | 依据 |
|---|---|---|
| TP | 1 | 单颗 48G 目标硬件 |
| 权重 | N128 W4A8；MTP 保留浮点权重 | 使用现有模型与已验收加载路径 |
| KV | 实验动态 C8 | 沿用112K候选，非官方310P C8后端 |
| MTP | 2 | 新GDN下MTP1/2对照已完成；8K Decode提升21.36%，四类chat提升1.66%–19.01% |
| Graph | FULL_DECODE_ONLY，mode0，capture1/2/3 | Graph占用约.31GiB |
| max_model_len | 114688 | 112 Ki tokens；实际检索输入114560+输出余量 |
| gpu_memory_utilization | .95 | 单芯片内存候选；非宣称.90一定不可用 |
| max_num_batched_tokens | 4096 | 比1024小幅提升Prefill，并恢复8K测试的输出和接受率表现 |
| 专家 chunk | 512 | 保持已测N128路径，只单独改变调度批量 |
| Prefix caching | 关闭 | 冷Prefill公平比较；prefix可选测试8K/32K TTFT约-83%，112K检索通过，但严格全文等价尚未确认 |
| max_num_seqs | 1 | 当前只验证单流，不能将数字当并发吞吐 |

## 性能与质量范围

8K/128三次中位：Prefill503.099、Decode37.100tok/s；32K/128：Prefill450.652、Decode29.174tok/s。原N128同日同协议基线分别391.591/37.012和357.577/29.045。相对Prefill提升28.48%/26.03%，Decode接近不变。不是与a1 W8A8 TP2的公平对照，也不是W4优于W8的证明。

输入是固定重复合成文本、每次不同前缀，prefix off，预热1次，重复3次；阶段速度由Prometheus请求阶段耗时计算，不是纯kernel。原基线是历史同日数据，未交替A/B。8K和32K六条输出全部与原N128对应文本一致。

GSM固定seed310/32题：31/32，与原N128相同，index205达到512tokens上限。小样本不证明广泛质量等价。8题基础生成与A/B/A缓存复用通过。启动KV容量186236tokens是池容量估计，不是最大单请求上下文。

4096候选实际114560token检索通过，输出14tokens，总114574tokens，耗时348.941s，较原N128的415.306s减少15.98%。单条检索不等于长文生成质量或最长上下文证明；其后8K三次复测完成：Prefill中位503.226tok/s、Decode37.079tok/s，三个输出均与长请求前对应输出一致，未观察到持续性能下降。结果见`results/pr17640-b4096-20261013/`。此前1024候选实际检索成功，耗时352.140s，不将其结果代替4096候选。

## 启动

从仓库根目录执行；复用现有隔离构建制品，不修改宿主SDK。芯片0须独占，先显式停止同芯片旧服务。

```bash
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
export MODEL_DIR=$release/model
export OPP_DIR=$release/n128-candidate/artifacts/opp/vendors/custom_transformer
export EXTENSION=$release/clean-build/artifacts/qwen36_int4_custom.so
export C8_LIBRARY=$release/c8-device-length-build/output/libqwen36_c8.so
export GDN_OPP_DIR=$work/pr17640-opp/vendors/custom_transformer
export GDN_LIBRARY=$work/gdn-extension/pr17640_gdn.so
export GDN_SOURCE=$work/pr17640/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py
bash experimental/run-gdn-112k-product.sh
```

外部PR来源和构建方法见 `docs/huawei-followup-20261013.md`。该脚本默认MTP2、batch4096、expert chunk512；关闭GDN开关的原启动器仍可用于回归。

## 后续边界

这轮收敛目标是已有112K单流候选。已完成新GDN下MTP1/2固定输入对照，prefix初测见results/pr17640-prefix-20261013；尚需MTP0对照、prefix更多质量/压力测试、并发2/4/8、TTFT/TPOT尾延迟、更多任务精度、长文decode128以上稳定性。48G以外的32G硬件、TP2及官方main迁移不能由本轮结果外推。MindIEQ可选Prefill路径已完成独立服务性能及小子集门禁，后续新A8/阈值128/Gate量化共享结果见results/mindieq-shared-gate-20261014；当前作为长 Prefill 候选分流，短输入回退原W4，尚未无条件默认开启。

本轮工程结论：采用4096/512作为新的单流实验基线，服务保留a2的18024端口；仍保留原启动器用于回归。MindIEQ 已在交替五次及64K/近112K长请求中相对原始W4显示稳定的 Prefill 优势，但它只作为 `MIN_TOKENS=128` 的长 Prefill 候选分流，不能改变本页基础 W4 组合的回归性质，也未将小样本门禁写成正式产品验收。


E5共享Gate组合新增114560token检索通过、297.832秒、后续sanity5/5；追加114560输入+128输出持续生成三次正在执行。不能把canary或vLLM缓存估计作为持续速度/最大上下文证明。
