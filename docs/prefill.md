# TP1 W4 Prefill 调优记录

目标是改善真实专家矩阵的大 M 路径，同时保留 Decode 收益及精度。历史 W8A8 约1000tok/s来自用户观测；a1当前为TP2、batch4096。尚无同机同TP同输入的严格W8/W4对照，不承诺TP1达到该值。

## 固定条件与两个分块参数

TP1、公开W4 checkpoint、FP16 KV、MTP=0、eager、max_model_len4096、memory utilization .90、max_num_seqs1、prefix off。同一3072-token prompt，输出1token，1次warmup、3次测量，使用`tools/bench_server_phases.py`服务端请求阶段计时；包含阶段调度与首token处理，非纯kernel时间。每次counter增量验证为1，保存prompt IDs和HTTP响应。

`MAX_BATCHED_TOKENS`控制一次模型forward的调度token数；`QWEN36_GROUPED_CHUNK_TOKENS`控制每层专家的内部切片。batch256时，专家chunk512/640实际不会扩大输入。专家chunk上限由5120routes/top_k8确定为640，不能直接将chunk设为1024/4096；大调度batch可由多个小专家chunk处理。

| 调度batch | 专家chunk | 三次Prefill tok/s | 中位数 | 状态 |
| --- | --- | --- | --- | --- |
| 256 | 256 | 324.6767 / 324.2034 / 322.4247 | 324.2034 | 实际请求计时通过，检索检查待补 |
| 1024 | 256 | 354.4466 / 355.0568 / 354.4582 | 354.4582 | 计时与3072-token中间needle通过 |
| 1024 | 512 | 387.7657 / 387.8843 / 387.8627 | 387.8627 | 计时与3072-token中间needle通过 |
| 1024 | 640 | 386.9475 / 387.0152 / 386.7189 | 386.9475 | 计时与3072-token中间needle通过 |
| 4096 | 512 | 384.3657 / 383.8421 / 383.9090 | 383.9090 | 计时与3072-token中间needle通过 |

两轮prompt IDs逐项一致（3072），第二轮Prefill中位数提高19.6356%。调度batch与专家chunk同时改变，尚不能独立归因；均未开C8/MTP/Graph。第二轮needle输出14tokens、正确返回ASCEND732941，耗时8.79984s；只有单个检索样本，不是正式准确率评测。batch1024时权重23.82GiB、peak activation2.75GiB、non-torch约.18GiB；这是启动profile预算，非请求实际峰值。

补充batch1024/chunk256对照后，三组prompt IDs再次逐项核验一致。固定chunk256，调度batch256→1024，中位吞吐+9.3320%；固定batch1024，chunk256→512，中位+9.4241%。这些是顺序跨服务三轮观察，尚无交替重复或尾延迟门禁。chunk256检索正确返回验证码，14输出tokens、9.68224s。raw `qwen36-prefill-b1024-g256-p3072-o1.jsonl`、`qwen36-prefill-b1024-g256-needle.jsonl`；640专家分块候选已启动，4096调度batch候选尚未测完。

640分块随后完成：相同prompt IDs，Prefill中位386.9475tok/s，相比512的387.8627下降约0.24%，三轮不足以认定显著差异，未显示扩大分块的收益。3072-token中间needle返回正确验证码，14输出tokens、8.80470s。raw `qwen36-prefill-b1024-g640-p3072-o1.jsonl`与`*-needle.jsonl`。因此batch1024下暂保留chunk512；下一组batch4096/chunk512已开始加载。

batch4096/chunk512已完成：prompt IDs与1024/chunk512逐项一致，三次Prefill中位383.9090tok/s，较387.8627下降约1.02%，没有显示继续扩大调度batch的收益。单needle检索通过，14输出tokens、8.88461s。raw `qwen36-prefill-b4096-g512-p3072-o1.jsonl`与`*-needle.jsonl`。最后batch4096/chunk640继续扫描中；当前较好参数仍为1024/512。达到约388tok/s后需profile定位专家pack/MatMul、dispatch和非专家层，不能只靠扩大batch期待1000tok/s。

第一轮不含MTP的权重实测23.8202GiB，峰值激活2.75GiB，non-torch约.19GiB；KV预算4.95GiB（310P worker日志明确halved for workspace）。此前MTP配置加载27.288GiB，不能将其作为关闭MTP配置的基础内存。32G实际业务容量仍需独立实卡或受控预算验证。

原始第一轮文件：`w4-prefill-256.log`、`w4-prefill-256-p3072-o1.jsonl`，保留在实验目录，不提交Git。第二轮日志`w4-prefill-b1024-g512.log`、请求`w4-prefill-b1024-g512-p3072-o1.jsonl`与`w4-prefill-b1024-g512-needle.jsonl`。代码提交eac6c0e引入专家chunk可选参数，不修改默认512。后续扫描已启动，依次1024:256、1024:640、4096:512、4096:640；结果写入新目录prefill-sweep-20261012。尚未得到结果，不宣称进一步收益。

## 扫描复现

`experimental/run-prefill-sweep.sh`顺序运行实验服务，每组计时后单独执行3072-token中间needle。需要独占davinci0、端口未被其他服务占用，设置已验证制品的绝对路径；不覆盖已有结果。

```bash
MODEL_DIR=/absolute/path/weights \
OPP_DIR=/absolute/path/build/artifacts/opp/vendors/custom_transformer \
EXTENSION=/absolute/path/build/artifacts/qwen36_int4_custom.so \
C8_LIBRARY=/absolute/path/c8-build/output/libqwen36_c8.so \
RESULT_DIR=/absolute/path/new-prefill-results PORT=18020 \
PAIRS='1024:256 1024:512 1024:640 4096:512 4096:640' \
bash experimental/run-prefill-sweep.sh
```

C8库仅为沿用实验启动器的必需挂载；`QWEN36_C8_ENABLED=0`，cache为FP16。每组服务名仍沿用`qwen36-w4a8-c8`，不是C8启用证据。超过 readiness 观察期限时脚本保留活容器并报错，不能据此认定运行终止或重新启动。

通过分块扫描后，按profile决定融合dispatch/count、SwiGLU/activation pack、优化expert weight复用或扩大原生route上限；之后回归MTP1，另测MTP2，再接入Graph/C8。仅参数变化不代表kernel效率改善。

## 后续MTP2设备门禁

标准启动器保持`MTP=0/1`作为开关；`MTP=1 NUM_SPECULATIVE_TOKENS=2`选择两次草稿预测，默认草稿数仍为1。实验C8启动器仅在`QWEN36_C8_ENABLED=0`时接受两草稿，C8后端的单草稿限制不变。先测试FP16 KV/eager，再评估Graph/capture形状及C8回退；当前没有MTP2设备运行结果。

固定镜像源码`vllm_ascend/spec_decode/llm_base_proposer.py`在单草稿early exit后具有`num_speculative_tokens-1`次迭代，并更新位置、hidden states和slot mapping。源码路径存在不代表310P的Qwen hybrid state回退已经正确，须记录第二位置接受率、实际拒绝、质量、平均接受长度、同协议TPOT和内存。不要仅以“能启动”宣布MTP2适配完成。
