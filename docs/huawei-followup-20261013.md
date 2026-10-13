# 华为会议后：MindIEQ QMM 与 GDN PR 独立实验

2026-10-13，a2 / Ascend 310P3，CANN 9.1.0，torch 2.10 / torch-npu 2.10，芯片 0。实验不覆盖已发布模型或宿主 SDK。停止旧 `qwen36-n128-quality` 服务后运行测试。

## 固定来源与计算路径

- [MindIEQ ops-nn](https://gitcode.com/mindIEQ/ops-nn/tree/fake_quant_matmul_x2nz_3_r3_biased)，固定提交 `d9c9871aa335f7b5638e9a2679a45bb3e7561d3a`。
- [GDN PR #17640](https://github.com/vllm-project/vllm-ascend/pull/17640)，固定 head `a2d228c7850f1749a4a934cc7514cd6c01fd7948`，base `75dc5fd3d230aba3c49c8ef74d1e6207b59bfb75`。
- Catlass 固定 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。

MindIEQ 分支不等同于现有 `QwenW4A8Int4MatmulV310`：权重以 biased nibble `(W+8)&15` 存储，设备 UB 解包后生成 L1 INT8 NZ，执行 INT8 Cube，并通过附加纠偏计算减去 `8*sum(A)`。这是 W4 存储、片上 INT8 计算路线，不是现有两路 signed INT4 Cube 的激活高低位组合。其接口限定对称 per-channel 权重 scale、per-token 激活 scale、groupSize=0；无法直接假定支持独立 G128 scale、非零 zero-point 或完整 MoE 调度。

PR #17640 改造 GDN 的 WY、FwdH、FwdO，并处理数值稳定性、因果 mask、尾块与流水线数据。Qwen3.6 配置的 key/value head dim=128、key heads=16、value heads=32 符合 WY 契约。PR 自述的模型/TP/批量不等同于本项目 TP1 35B；不引用其倍数作为本项目端到端收益。

ops-nn/CANN 算子文件使用 CANN Open Software License V2；本仓库保留外部源码引用与测试代码，不将其整体复制成 Apache-2.0 代码。

## 已验证：MindIEQ QMM

隔离编译成功。首次仅设置外置 `ASCEND_CUSTOM_OPP_PATH`，调用失败于 `QuantBatchMatmulV3` tiling compile info parse，错误 561103，未进入矩阵计算。按作者 README 注册容器内 OPP vendor、`config.ini load_priority=custom_nn` 后成功。宿主 SDK 未修改。

作者原始测试：

- `(M,K,N)=(1,2048,2048)、(16,2048,2048)、(1024,2048,2048)` 均 mismatch=0，容差 `atol=rtol=0.01`。
- `(16,2048,2048)` 和 `(1024,2048,2048)` ACL Graph 捕获成功、8 次回放逐位一致。固定输入和地址测试，尚不代表动态输入/完整 vLLM Graph 兼容。
- 20 次同步主机测量：M16 direct 107.0 μs / replay 81.1 μs；M1 direct 104.8 μs / replay 78.9 μs。不是纯 Kernel 延迟。

补充测试：gate/up 合并 K2048/N1024，down K512/N2048；M=1/2/4/8/128/512/1024/4096，INT8 激活覆盖 [-128,127]，所有 16 组 `rtol=atol=.002` 通过，最大绝对误差 <.001。权重仍为随机整数，尚未提取真实 checkpoint 专家。

权重纯数据字节减半：gate/up 2 MiB→1 MiB，down 1 MiB→.5 MiB，不含 scale、工作区和模型其余部分。

同整数/scale 的 W8 对照需要先将转置权重转 FRACTAL_NZ，再取转置视图。初次 ND 调用错误 161002，日志确认 storage format 检查失败；不能将这次失败当作 W8 性能结果。下表来自修正后的 NZ 测试，每组 warmup=5、测量=30，按 W4 后 W8 顺序，无 Graph，包含主机分发、launch、同步，排除离线打包。

| 矩阵 | M | W4 μs | W8 μs | W8/W4 |
|---|---:|---:|---:|---:|
| gate/up | 1 | 93.5 | 73.5 | 0.79 |
| gate/up | 128 | 112.5 | 83.5 | 0.74 |
| gate/up | 1024 | 182.6 | 151.5 | 0.83 |
| gate/up | 4096 | 593.8 | 518.2 | 0.87 |
| down | 1 | 78.3 | 74.0 | 0.94 |
| down | 128 | 86.4 | 102.7 | 1.19 |
| down | 1024 | 167.5 | 141.8 | 0.85 |
| down | 4096 | 655.5 | 569.0 | 0.87 |

大多数形状 W4 较慢，单个 down/M128 较快不足以建立可重复加速结论。也未与本项目现有 native W4 GMM 做同条件对照。下一步应比较 Graph 下的双方及真实权重/多专家路由；不要据此替换模型后端。

## 复现构建

a2 工作目录 `/home/ychy/ascend-lab-runtime/mindieq-r3-20261013`，下列 `/work` 为其容器挂载点。镜像来源 `quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`；仅容器补装 `dos2unix pigz`、禁用损坏 Triton 目录。

```bash
# 外部目录；固定 revision
 git clone -b fake_quant_matmul_x2nz_3_r3_biased https://gitcode.com/mindIEQ/ops-nn.git ops-nn
 git -C ops-nn checkout d9c9871aa335f7b5638e9a2679a45bb3e7561d3a
# 以下在隔离容器运行
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/ops-nn
bash build.sh --pkg --ops=quant_batch_matmul_v4 --soc=ascend310p -j8
bash build_out/cann-ops-nn-custom_linux-x86_64.run --install-path=/work/opp
```

扩展构建使用外部 `matmul/quant_batch_matmul_v4/tools/torch_extension` 的独立副本 `extension-source`。仅修改 CMake：`aarch64-linux`→`x86_64-linux`；`VENDOR_LIB=/work/ops-nn/build`；`VENDOR_INC=/work/ops-nn/matmul/quant_batch_matmul_v4/op_host/op_api`；移除 include 后缀 `/aclnnop`。

```bash
cmake -S /work/extension-source -B /work/extension-build -DCMAKE_BUILD_TYPE=Release
cmake --build /work/extension-build -j4
# 仅容器内注册内嵌 vendor
cp -a /work/opp/vendors/custom_nn "$ASCEND_HOME_PATH/opp/vendors/custom_nn"
printf 'load_priority=custom_nn\n' > "$ASCEND_HOME_PATH/opp/vendors/config.ini"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
# test-source 为作者测试副本，仅替换 library 路径；环境指定 visible device=0
python /work/test-source/test_qmm_v5_ext.py
python /work/test-source/test_qmm_v5_ext_graph.py
```

作者测试仅打印失败而不保证非零退出码，必须检查 `ALL PASSED` 与 mismatch。补充测试使用 assert、失败会非零退出。

```bash
WORK_DIR=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013 \
  bash experimental/run-mindieq-qmm-check.sh
```

该 runner 使用已登记 vendor 的本机派生镜像 `localhost/qwen36-mindieq-embedded:20261013`，只挂载芯片 0。原始日志与全部样本：`results/mindieq-r3-20261013/`。

## GDN 实验进度

固定 PR archive 独立解包到 `/work/pr17640`。首次编译漏设 Catlass，保留失败日志；补齐锁定依赖后三个算子编译、打包成功。WY 独立 Torch 扩展已编译，使用 `pr17640_wy` namespace，不与官方 `_C_ascend` 冲突；绑定提取保留原版权与许可。

```bash
export CPATH=/work/pr17640/csrc/third_party/catlass/include
cd /work/pr17640/csrc
bash build.sh --pkg \
 --ops='chunk_gated_delta_rule_compute_wy;chunk_gated_delta_rule_fwd_h_vllm;chunk_fwd_o_vllm' \
 --soc=ascend310p -j8
python /project/tools/build_pr17640_wy_extension.py \
 --source-root /work/pr17640 --build /work/wy-extension
```

WY 与原 torch WY 的数值、延迟测试已启动，首次 fallback 触发 CANN 动态编译。128 tokens 的两组先通过 CPU 参考：输出 q/k 精确一致，w/u 最大绝对误差 ≤7.63e-6，g 最大绝对误差 ≤9.54e-6；测试容差 w/u/g 为 rtol=.03、atol=.0002。小衰减时 WY 0.314 ms 对照 9.386 ms，大衰减时 WY 0.233 ms 对照 9.355 ms（5 次 warmup、20 次同步调用）。这仅是 WY 子步骤收益。512/1024 tokens 继续运行中。随后仍需验证 FwdH/FwdO 和真实模型端到端。不将编译成功写成推理成功。

OPP 包 SHA256：`070f2e65f0ce9c82da98d7fc94de1a9738f47126dc9b9b33edbd99c9ae9edcd2`。QMM 扩展 SHA256：`6e59fcbd358ad958ba84a53d3e58bd2fa196ec65959688c5b7bca7a10252ff3f`。Graph debug JSON 仅用于验证 task 名称（实际命中 `QuantBatchMatmulV3_NZ_NZ_int8_int4_fp16`），其中 duration 不作为 profiler 性能数据。

WY 测试命令（同一隔离容器，先安装 PR OPP、设置 vendor 环境）：

```bash
python /project/tools/check_pr17640_wy_310.py \
 --source /work/pr17640/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py \
 --library /work/wy-extension/pr17640_wy.so --out /work/pr17640-wy.jsonl
```

CPU 参考直接提取固定 PR 的原始 torch WY 函数，不复写其数学逻辑；相同函数在 NPU 上作为速度对照。数值用 CPU 比对，速度 warmup 后测量，避免把首次 JIT 编译算入延迟。
