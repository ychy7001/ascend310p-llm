# MindIEQ 完整 MLP 与专家占用门禁

2026-10-13，a2芯片0、310P3/CANN9.1/Torch2.10；独占算子测试，非模型吞吐。使用真实layer0/expert0..7权重，合成FP16输入；包含激活量化、Gate/Up、FP32 SwiGLU→FP16、Down、Python循环与输出拼接。固定分组，排除router、排序/route combine、共享专家与Attention。native小于等于128行用融合SwiGLU pack，大于128行用FP32 SwiGLU和pack。

## 发现与修正

原始外部绑定直接调用ACL，以当前stream(false)提交，没有通过torch-npu命令队列。动态量化生产者接入后Down出现非有限输出（v5日志）。转FP32 scale、ND连续布局未解决；调用前同步后数值恢复正常。这支持队列顺序问题判断，非证明硬件不能计算。

隔离补丁工具patch_mindieq_queue_binding.py生成新外部源码，使用OpCommand::RunOpApiV2提交，输入/输出由闭包持有；不改原始外部文件。新绑定编译成功。去掉调用前同步后八expert10组全部通过，256槽位4组也通过。这里仍是实验绑定，不是已上游审查的生命周期实现；Graph和动态多stream尚未验收。

激活使用native fused pack或npu_dynamic_quant两个实现，不能宣称完全相同整数。完整MLP相对native的L2误差约1.0%–1.1%，小样本finite与L2<3%门禁通过；该阈值仅诊断，非模型精度验收。MindIEQ与W8输出误差相同，提示共同动态量化路径差异，仍需统一舍入后质量验证。精确整数单矩阵精度另见w4-three-backends-20261013。

## 完整MLP中位ms

预热5、重复20、三后端交替顺序；同步主机计时，非纯kernel。原始jsonl全部保存。

| 路由条数 / 活跃槽位 | 当前W4 | MindIEQ W4 | W8 |
|---|---:|---:|---:|
| 128 / 8均分 | .477 | 1.085 | .943 |
| 512 / 8均分 | 1.717 | 1.210 | 1.072 |
| 1024 / 8均分 | 2.779 | 1.392 | 1.119 |
| 4096 / 8均分 | 11.318 | 2.724 | 2.365 |
| 512 / 256均分 | 11.000 | 23.441 | 21.425 |
| 4096 / 256均分 | 21.720 | 25.788 | 21.950 |
| 4096 / 1集中（256槽位） | 13.465 | 2.560 | 2.302 |

模型配置实际num_experts256/topk8；512token chunk约4096路由。256槽位实验将真实前8专家权重重复32次，用于合成专家占用压力，不是256个不同真实专家，不是动态router或真实分布采样。每槽2/16行时，逐专家调用开销吞掉大矩阵收益；不会将8专家均分的约4倍外推到模型。

门禁决策：逐专家Python方案暂不进入模型。保留当前W4和已测配置。下一步价值在设备端grouped/GMM，把偏置INT4解包/INT8计算融合进单次多专家调度，并保持设备group_ends；或从真实路由trace证明某层的大expert负载足以支付混合调度成本。当前不能按全局token数直接选MindIEQ。

## 复现

先按huawei-followup-20261013.md取得固定外部源与隔离镜像。在新目录生成并编译队列绑定（CMakeLists使用外部原版）：

```bash
python3 tools/patch_mindieq_queue_binding.py --source /work/extension-source/qmm_v5_ext.cpp \
  --output /work/queue-extension-source/qmm_v5_ext.cpp
cmake -S /work/queue-extension-source -B /work/queue-extension-build -DCMAKE_BUILD_TYPE=Release
cmake --build /work/queue-extension-build -j4
```

外部test helper仅把加载库路径改为新queue-extension-build，保持packing函数。实际容器、挂载与运行参数见run-full-mlp-queue.sh和run-full-mlp-e256.sh：

```bash
python3 tools/bench_mindieq_mlp_310.py --model /model --library /custom/qwen36_int4_custom.so \
  --upstream-test /work/test-source/test_qmm_v5_queue.py --out /work/mlp8.jsonl
python3 tools/bench_mindieq_mlp_310.py --model /model --library /custom/qwen36_int4_custom.so \
  --upstream-test /work/test-source/test_qmm_v5_queue.py --experts 256 --rows 512 4096 --out /work/mlp256.jsonl
```

sync-full记录为中间诊断，含每expert显式同步/数值探针，不能与最终queue计时直接归因加速。原始失败与隔离构建日志保留，未对外发布修改后的CANN源或新权重。

隔离队列绑定so SHA256：741bc86867e42b3d068e0797ca2a4dd730f0b3de1be070cb25671a658b7ffdfa。未包含该二进制到Git。

统一run-candidate入口服务启动并完成生成sanity5/5；8K/128三次Prefill中位507.211、Decode36.944tok/s，prompt hashes全部与原基线匹配，输出3/3全文匹配。服务保留18024，prefix关闭。该回归不替代完整质量/长文测试，也不是MindIEQ模型集成。
