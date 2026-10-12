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
| 4096 | 640 | 391.8055 / 392.0145 / 391.4968 | 391.8055 | 计时与3072-token中间needle通过 |

两轮prompt IDs逐项一致（3072），第二轮Prefill中位数提高19.6356%。调度batch与专家chunk同时改变，尚不能独立归因；均未开C8/MTP/Graph。第二轮needle输出14tokens、正确返回ASCEND732941，耗时8.79984s；只有单个检索样本，不是正式准确率评测。batch1024时权重23.82GiB、peak activation2.75GiB、non-torch约.18GiB；这是启动profile预算，非请求实际峰值。

补充batch1024/chunk256对照后，三组prompt IDs再次逐项核验一致。固定chunk256，调度batch256→1024，中位吞吐+9.3320%；固定batch1024，chunk256→512，中位+9.4241%。这些是顺序跨服务三轮观察，尚无交替重复或尾延迟门禁。chunk256检索正确返回验证码，14输出tokens、9.68224s。raw `qwen36-prefill-b1024-g256-p3072-o1.jsonl`、`qwen36-prefill-b1024-g256-needle.jsonl`；后续候选结果见下文。

640分块随后完成：相同prompt IDs，Prefill中位386.9475tok/s，相比512的387.8627下降约0.24%，三轮不足以认定显著差异，未显示扩大分块的收益。3072-token中间needle返回正确验证码，14输出tokens、8.80470s。raw `qwen36-prefill-b1024-g640-p3072-o1.jsonl`与`*-needle.jsonl`。因此batch1024下暂保留chunk512；batch4096结果见下文。

batch4096/chunk512已完成：prompt IDs与1024/chunk512逐项一致，三次Prefill中位383.9090tok/s，较387.8627下降约1.02%，没有显示继续扩大调度batch的收益。单needle检索通过，14输出tokens、8.88461s。raw `qwen36-prefill-b4096-g512-p3072-o1.jsonl`与`*-needle.jsonl`。batch4096/chunk640结果见下文。达到约388tok/s后需profile定位专家pack/MatMul、dispatch和非专家层，不能只靠扩大batch期待1000tok/s。

完整扫描已完成，进程2003800正常结束、实验容器已停止。4096/chunk640的相同3072-token输入中位391.8055tok/s，为当前扫描最高值，较起点324.2034提高20.8517%，较1024/512提高1.01654%。后一个差距尚未做交替复测，不能认定稳定显著。单needle正确返回验证码，14输出tokens、8.68919s。raw `qwen36-prefill-b4096-g640-p3072-o1.jsonl`及`*-needle.jsonl`。短输入只有一次4096调度，不代表长输入chunked历史读取的效果；已开始32K服务4096/640长输入候选，计划8K/约32K与1024/512对照。

## 请求级NPU Profile

`PROFILE_DIR=/absolute/path/new-profile-dir`可显式启用实验启动器的torch profiler配置，默认关闭。镜像使用torch_npu CPU/NPU活动、Level1、PipeUtilization，输出挂载到宿主指定目录。采集前先warmup，`tools/profile_http_prefill.py`调用`/start_profile`，执行一次固定prompt/输出1的请求，再`/stop_profile`。采集请求的HTTP耗时不用于常规吞吐结论；kernel累计时间也不能直接解释为不重叠墙钟占比。该入口已在真实310P3服务完成一次3072-token采集，输出166MB profiler制品；详细分析待完成，采集期间请求耗时不作为常规吞吐。

第一轮不含MTP的权重实测23.8202GiB，峰值激活2.75GiB，non-torch约.19GiB；KV预算4.95GiB（310P worker日志明确halved for workspace）。此前MTP配置加载27.288GiB，不能将其作为关闭MTP配置的基础内存。32G实际业务容量仍需独立实卡或受控预算验证。

原始第一轮文件：`w4-prefill-256.log`、`w4-prefill-256-p3072-o1.jsonl`，保留在实验目录，不提交Git。第二轮日志`w4-prefill-b1024-g512.log`、请求`w4-prefill-b1024-g512-p3072-o1.jsonl`与`w4-prefill-b1024-g512-needle.jsonl`。代码提交eac6c0e引入专家chunk可选参数，不修改默认512。后续扫描依次完成1024:256、1024:640、4096:512、4096:640；结果保存在新目录prefill-sweep-20261012，汇总见上表。

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

## 长输入与产品参数门禁（2026-10-12）

4096/640的32K服务测量已完成并正常停止：TP1、FP16 KV、MTP=0、eager、memory .90、batch4096/chunk640、max seq1、prefix off。8K输入输出1token、warmup1后三轮Prefill为384.2473/385.2984/383.9917tok/s，中位384.2473；HTTP TTFT为21.518/21.394/21.623秒。近32K输入32256token三轮Prefill为358.5912/358.0136/357.3548tok/s，中位358.0136；TTFT为90.365/90.290/90.423秒。近32K中间needle检查通过，usage输入32256+输出14=32270token，正确返回ASCEND732941，耗时94.0191秒；单样本不代表完整长上下文精度。随后3072-token NPU profile采集成功。长输入后输出256token的持续Decode结果见下文。原始文件`w4-long-b4096-g640-p8192-o1.jsonl`和`w4-long-b4096-g640-p32256-o1.jsonl`。

产品参数分别报告：配置上限、实际请求验证长度（输入+输出）、业务推荐长度、KV物理容量。不能相互替代。长度档位先覆盖8K/32K，再对候选组合测试64K/128K，若内存和质量允许再到原生262144上限；不以KV估算承诺300K。每档需要Prefill tok/s和TTFT、长输入后固定输出256token的Decode tok/s和TPOT、峰值内存和质量检查。正式P95需要足够请求样本，三轮观察只作为探索，不报告可靠P95。

先在相同32K上限下对比4096/640与1024/512（8K和32256token固定IDs）。输出1token隔离Prefill趋势，输出256token观察持续Decode；检索样本单独运行。之后回归MTP1/2，再结合C8与Graph完成组合门禁。32G规格还需要实际32G设备或受控预算验证；48G设备上的成功不能替代32G结论。

真实profile初步统计：3072token/output1，400次`QwenW4A8Int4MatmulV310`累计3.980758秒，单次5.2924～15.3176ms，op_statistic中占累计时间50.402%；其次Slice .422909秒、Mul .361105秒、ReduceSum .291646秒、ViewCopy .279497秒、普通MatMulV2 .266708秒、Tril .262126秒、Cast .231858秒。step_trace的Computing为7.840336秒、Stage7.895287秒，无通信。累计算子比例不直接等于普通请求墙钟占比。`kernel_details`同时给出cube_utilization约93.87%和mac_ratio约.0325，两者口径不同，当前不能用前者声称Cube已达峰值或用后者认定硬件算力利用率3%；没有实测GB/s，暂不认定memory/compute bound。原始目录`prefill-profile-b4096-g640`，本地摘要`w4-prefill-profile-b4096-g640-summary.json`。

8K输入/输出256token三轮有效请求：Prefill 384.8600/384.9462/384.5370tok/s；Decode 21.0647/21.0749/21.0600tok/s，中位21.0647。Decode分母为服务端decode阶段耗时，分子输出token数减1；HTTP TTFT中位21.3138秒，请求平均TPOT中位47.4792ms。这些是FP16KV/MTP0/eager配置，不能称为W4+MTP+C8+Graph组合结果。近32K持续Decode结果见下文。

长输入对照可用`experimental/run-long-context-sweep.sh`复现，沿用前述MODEL_DIR/OPP_DIR/EXTENSION/C8_LIBRARY，设置全新的RESULT_DIR；默认顺序比较1024:512与4096:640，固定32K上限、8K/32256输入、1/256输出、warmup1+三轮，最后执行32256中间needle。每组正常通过后才停止服务进入下一组；失败保留服务和原始证据。4096/640测量完成后，已经正常启动`PAIRS=1024:512`，结果目录`long-context-sweep-20261012`。

近32K输入32256/输出256token三轮有效请求：Prefill 358.0764/358.1915/358.4693tok/s；Decode 18.8163/18.7997/18.8091tok/s，中位18.8091。HTTP TTFT中位90.1193秒，请求平均TPOT中位53.1154ms。同配置8K→近32K，Decode中位下降10.7079%，说明历史长度有可测影响，不能据此推断MTP/C8/Graph组合幅度相同。原始文件`w4-long-b4096-g640-p32256-o256.jsonl`。32K输入持续生成完成，没有OOM或使用量不符；固定重复文本的吞吐测试不代表正式生成准确率。
