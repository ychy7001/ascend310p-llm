# Qwen3.6-35B-A3B：Ascend 310P 适配与优化评审

2026年10月13日 · 技术会议版 v1 · 实验快照 609974f

## 1. 会议结论与讨论目标

目标是在单颗310P3、48GB级NPU内存上，以W4A8 routed experts、MTP和INT8 KV支撑超过100K上下文，兼顾Prefill、Decode与精度，并将实现收敛成可维护、可评审的主线增量。32GB、256K完整请求与TP2 C8尚未验收。

目前已实现TP1 W4A8+C8+MTP2+Decode Graph启动、短请求与112K近上限检索。最新N=128专家Prefill tile在同配置顺序对照中提升约6.5%～7.4%，Decode基本持平。当前最主要性能差距仍是冷Prefill，而非权重是否能加载。

本次希望明确三件事：①310P整数Cube路径与更高效的W4 Prefill建议；②是否有可公开的INT8 KV Attention接口，避免历史KV转换；③310P Graph生命周期的正式修复边界。希望获得接口、示例、tiling建议或可复现诊断方法。

当前环境：a2第一颗Ascend310P3，TP1；镜像v0.23.0.post1-310p，CANN9.1，Torch2.10，torch-npu2.10.0.post4；npu-smi驱动报告26.1.1。标称48GB级板载内存，驱动报告可用容量约43GiB，容量测算按实机值。a1的TP2 W8A8仅为方向性参考。

代码：https://github.com/ychy7001/ascend310p-qwen36

权重：https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P

已验证的是text-only生成及有限数值/检索门禁，不代表全面精度、多模态、稳定并发或正式P95通过。会议前运行的32题固定GSM8K子集结果须以完整原始记录为准，本版不预填成绩。

## 2. 整体计算与权重适配路径

权重：Qwen3.6原始BF16 → ModelSlim SSZ专家INT4 → checkpoint元数据/打包 → 独立加载。仅40层routed experts使用W4；其余张量保留浮点，runtime使用FP16执行策略。Qwen3.6自身19个MTP张量保留BF16制品，不借用Qwen3.5 MTP权重。

当前量化是对称per-output-channel SSZ；每通道scale按G128算子ABI重复存放，offset为零。G128布局不等于每128元素独立拟合scale。打包按输入轴低nibble优先。转换来源、制品哈希和模型索引见docs/provenance.md及发布校验；源BF16不可变revision尚未完整追溯。

MoE：原router → 按专家排序 → Gate/Up W4矩阵 → SiLU×Up → Down W4矩阵 → 路由加权/归约。Decode保留小行数路径；Prefill使用分组批量路径。TP1无HCCL；TP2专家所有权及共享专家切分单独处理，不能将TP2通信profile用于解释TP1瓶颈。

W4A8：设备端per-token激活量化为INT8，再拆为INT4 low/high limbs，组合整数Cube乘积与修正项；权重保持INT4打包。源码使用int4b_t/Mmad并已在310P执行，不是每token把完整W4权重反量化到全局FP16；也不宣称单条原生INT8×INT4指令。

C8：full-attention的K/V写INT8、scale用FP16；读取历史页时转换为FP16 NZ工作区，复用310P原生PA/split-fuse。linear-attention状态保持独立。split-by-type池与共享scratch降低占用，但转换和padding使实际节省小于简单减半。

MTP与Graph：复用现有Qwen3.5/Qwen3.6 draft/verify机制，保持MTP decoder走浮点路径；Decode使用FULL_DECODE_ONLY，mode=0，MTP1捕获[1,2]，MTP2捕获[1,2,3]。Graph中长度等元数据在设备端更新，避免capture时非法CPU同步。

## 3. 与主线的改动边界及收敛方案

当前是固定官方镜像上加载独立扩展、社区模块子集和opt-in适配层，并非已在官方main上验证的最小补丁集。未修改宿主生产安装不等于零行为改动；monkeypatch会改变隔离进程中的模型构造、加载和cache生命周期。

算子边界：native-overlay/为AscendC源码快照；tools/qwen36_int4_custom.cpp注册独立qwen36_int4 namespace，避免覆盖官方_C_ascend。拟保留pack/matmul/SwiGLU必要接口，未通过广泛门禁的down-reduce等候选不默认启用。

模型边界：runtime/.../qwen36_w4_patch.py约239行，按ascend_expert_quantization元数据替换routed MoE、读取sidecar、处理TP所有权，并排除MTP decoder。当前临时替换构造器中的MoE类以避免先分配BF16专家；上游化应移入明确量化后端/模型构造钩子，消除全局临时替换。

社区代码边界：runtime/.../qwen4_exp/保留执行所需模块子集；来源opensensor提交0637d3baa6bc2000ba6433b301f233c6516847b4。保留SPDX、版权与来源，逐文件审查依赖，移除无关模型映射和重复后端；不能整体合并社区fork。

C8边界：experimental/c8_vllm_adapter.py约275行，修改get_spec、get_configs、allocate、forward四类入口。上游化应形成独立cache spec/backend和显式allocator契约；把payload、scale、hybrid state与scratch计入预算，消除engine/cache工具的双入口monkeypatch。

Graph边界：experimental/direct_graph_310.py约32行，包含_use_aclgraph兼容及use_updatable_graph全局lambda诊断替换。后者对所有attention返回true，范围过宽；应按固定版本的310P后端身份和生命周期收敛，逐一验证target/draft图，不以跳过更新掩盖错误。

推荐拆分顺序：算子与数值/形状测试 → 权重元数据和专家加载 → Qwen3.6 routed MoE接入 → C8实验后端 → Graph版本修复。MTP复用框架，仅提交W4组合的必要兼容修改。main迁移应使用独立worktree并固定commit，保持现有镜像基线可复现。以上是待执行收敛方案，不宣称已形成上游PR。

## 4. 已测性能、上下文与参数取舍

产品性能候选：TP1；max_model_len=114688；gpu_memory_utilization=0.95；max_num_seqs=1；batch=1024；专家chunk=512；C8 fused-write/split-by-type/device-length；MTP2；Decode Graph mode=0、capture[1,2,3]；prefix cache关闭。

最新配对：8192输入/128输出，N64 Prefill/Decode=364.49/37.12 tok/s，N128=391.59/37.01，Prefill+7.43%；32256/128，N64=335.72/29.08，N128=357.58/29.05，Prefill+6.51%。每档warmup1、三轮中位数、独占端点；顺序配对，非随机交替、非P95。

112K容量：两者均完成114560输入检索，正确输出14 tokens；N64总435.70s、N128总415.31s。总时间包含Prefill/生成，不能当纯Prefill。启动报告KV容量186236 tokens仅为调度估算；实际验收上限仍是114688配置。C8池116 blocks、4336091136 bytes，scratch预留234881024 bytes，权重加载27.288GiB、Graph约0.31GiB。

独立算子：真实layer0专家0～7、synthetic激活；Gate/Up N1024/K2048、Down N2048/K512，rows129/257/1024，balanced/one-expert共12 case全部通过。最大绝对误差约1.23e-4/6.11e-5；参考是精确激活整数与已量化权重，不能替代原BF16模型精度评估。

C8对C16：32K匹配W4/MTP1/Graph测试中，8K Decode30.51对32.29，下降5.5%；近32K23.19对26.70，下降13.2%；Prefill差异小于1%，观测峰值少约2.5GB。C8优先解决内存余量，长历史转换可能拖慢Decode；C16应保留为速度对照/回退档。

MTP1对2：112K Graph、32256输入/256输出同协议测试，MTP1 Prefill/Decode339.28/23.20，MTP2=336.56/28.94；Decode+24.8%、Prefill-0.8%。重复文本有利于接受率，正式业务结果待补。maxseq1下并发2/4排队测试总输出约5.06 tok/s，不能用单流Decode代替并发总吞吐。

Prefix命中：另一2048 batch档测得8K冷约361 tok/s、热约2260 tok/s；112K同前缀请求耗时约412s→9.33s。命中不等于冷Prefill或Decode加速。hybrid模式有block/batch约束，应独立提供会议前缀复用档，不混入冷请求比较。

## 5. 性能瓶颈与希望华为协助的问题

真实TP1 profile：QwenW4A8Int4MatmulV310累计50.4%，Slice5.36%、Mul4.57%、ReduceSum3.69%、ViewCopy3.54%、MatMulV2 3.38%、Pack2.27%。这是累计算子时间比例，不是墙钟占比，也不足以定性memory-bound。TP1 profile没有HCCL/AllReduce项。

P0｜W4 Prefill：请确认310P3上int4b_t/Mmad的推荐形状、布局和吞吐解释，是否能减少low/high两次乘积及修正开销；对真实N1024/K2048、N2048/K512，建议怎样做L1/L0复用、双缓冲和tiling？现有N128约提升7%，距离a1 TP2 W8A8冷Prefill约1.3K仍有差距，但不能直接按芯片数线性归因。

P0｜C8 Attention：是否有适用于310P3、可公开使用的INT8 KV PA/Block Table接口或样例？若没有，是否支持按历史页流水反量化、与QK/PV融合以避免整段FP16 NZ scratch？请区分硬件能力、SDK公开接口和仅910B/A2可用能力。

P0｜Graph：固定镜像下为何310P attention选择到通用update_graph_params？正确的updatable graph身份、参数初始化与重放更新应放在哪一层？希望获得target/draft共存的最小示例。mode3 split-graph另有节点多用户错误，当前mode0只是可运行路线。

P1｜内存：hybrid linear state、full-attention页、C8 scales和MTP state的最小对齐与生命周期是什么？是否可减少当前padding和保守预算？能否获得物理带宽/AI Core计数的推荐采集方法，分离转换、Cube与launch耗时？

P1｜MTP2：310P上draft/verify capture形状、rollback和KV更新是否有推荐实现？如何在真实会议/问答负载统计接受率与尾延迟？希望以同一题目/输出预算的对照判断默认MTP1或2，而非只看有利重复文本。

建议会后先完成一个可验证小闭环：华为推荐一个W4 Prefill tile/流水或C8读取方案；我们提供固定形状、权重哈希、最小脚本和原始latency，先验证精度/收益，再进入服务。若路径只有内存收益而无速度收益，也如实记录适用边界。

## 6. 复现、证据索引与后续验收

固定快照：609974f。构建scripts/build-native.sh；标准W4/FP16 KV服务scripts/run-service.sh；112K组合experimental/run-c8-112k-product.sh。QWEN36_PROFILE=mtp1或mtp2显式选择档位。路径通过MODEL_DIR、OPP_DIR、EXTENSION、C8_LIBRARY传入；详细命令见README和对应results目录。

组合验收：RESULT_DIR指定新目录，BASE_URL指向独占端点，运行bash experimental/check-n128-product.sh。独立算子：MODEL_DIR/OPP_DIR/EXTENSION/RESULT_DIR/LABEL设置后运行bash experimental/run-w4-prefill-check.sh，需先停止同芯片服务；结果包括全部12个case和重复latency样本。

证据：results/n128-product-20261013/保存候选阶段计时、metrics、检索、启动日志与kernel检查；results/n128-product-baseline-n64-20261013/保存配对N64与比较JSON；results/tp1-prefill-profile-20261012/保存TP1算子统计。docs/parameter-selection-20261012.md维护参数历史与当前边界。

制品标识：N128内核源码SHA256=8727515fba385b37604faa35715d59d2374017d3105a05d71b99250f0f4e7350；C8库=85f383cfab4e1f5595394bb1317febfcc65b7b04815c55e8a39ab8b94c3f05e1；W4扩展=a8ac855968ddc3a56db676c1ac8dd9a8657dbae2fffb28c4276ddb7fef4b5950。

后续门禁：固定真实业务集合比较MTP0/1/2及C8/C16；相同输入/输出与冷/热前缀条件测TTFT/TPOT P50/P95、显存峰值；112K多位置/多样本文档检索和持续生成；TP2 C8与32GB需独立实测。不能用单次启动、toy8/8或一个needle代替全面验收。

来源：社区https://github.com/opensensor/vllm-ascend；官方主线https://github.com/vllm-project/vllm-ascend；官方Graph与310P源码路径见docs/graph-upstream-patch.md。本次通过GitHub API锁定官方main为d6ce55f96a01d7239f04e313d1b4a46b01c18dc9；公开main与固定镜像存在版本差异。本报告是接口级改动清单，不是已验证的main全量diff。

维护约定：报告源文件在本仓库docs/huawei-adaptation-review-20261013.md；后续原始证据写results/。本次会议DOCX/PDF按用户要求导出到/home/ychy/workspace/data/w4a8，不再更新此前data/ascend目录。会议版文件与仓库代码分别版本化，评审意见回填源报告。
