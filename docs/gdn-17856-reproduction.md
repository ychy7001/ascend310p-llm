# 310P GDN 上轮接受数大于当前 query 长度：最小复现

2026-10-14，a2 第二颗 Ascend 310P3，镜像
`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，CANN 9.1。
使用镜像内置完整 custom_transformer OPP；PR17640 的单独 OPP 包没有 recurrent 算子的 ACLNN API，不能替代完整 OPP。

## 已完成的验证

脚本：`tools/repro_gdn_17856.py`。原始日志：
`results/gdn-17856-20261014/operator-original.log`。

两请求当前长度 `[2,4]`，状态表原始两行各有四槽，当前压紧后的槽号为
`[0,1,4,5,6,7]`。NK=4，NV=8，K=V=128，FP16。
设置 k/v/beta=0、g=log(0.5)，使每个 token 的状态恰为上一步的一半。
各状态初始值 0.25，额外将 A 的历史第 4 槽置为 0.5，用于排除选错历史状态的假通过。

| 路径 | 上轮接受数 | A 当前两个槽期望 | A 当前两个槽实测 | 结果 |
| --- | --- | --- | --- | --- |
| 原始合法对照 | `[2,1]` | `[0.125,0.0625]` | `[0.125,0.0625]` | 通过 |
| 原始异常 | `[4,1]` | `[0.25,0.125]` | `[0.25,0.25]` | 未按递推更新 |
| 实验状态预取 | `[4,1]` | `[0.25,0.125]` | `[0.25,0.125]` | 最小算例通过 |

异常路径请求 B 的完整状态也与预期不符。输出缓冲有复用旧值的现象；不能单凭输出非零或生成接口未报错判定正确。
合法对照和状态预取路径的全部状态槽均与确定性预期一致，脚本有断言。

## 修复方向与边界

不能简单删除 `acceptedTokenNum > seqLen` 检查，也不能把接受数截断到当前长度。
压紧索引只保留当前 query 槽，上一轮第 4 个历史状态仍须从原始完整请求行中选择。

脚本的实验路径先把 A 第 4 个历史状态保存到当前基槽，再以 accepted=1 调用原算子。
这个写法只用于证明状态选择方向，**不是通用 wrapper 修复，也没有部署到服务**。
它尚未验证任意请求数量、状态别名、随机数据、Graph、padding、混合 batch 或真实生成。
后续实现应显式分离“每请求初始状态索引”和“当前 token 写回索引”，或在完整表仍可用时批量 gather 历史状态。
所有源状态须先 gather 再写，避免别名覆盖。

此次足以支持补充 Issue 的实机复现证据，但尚不足以宣称生产修复完成。
也未证明此前模型 MTP 接受率下降就是本问题导致。

## 复现环境

将脚本挂入独立容器，只暴露空闲的实验 NPU 与 manager/devmm/hisi_hdc 设备，
挂载主机 driver/lib64、driver/version.info、dcmi、ascend_install.info。
设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`ATEN_CPU_CAPABILITY=default`，
移开镜像内不兼容的 triton 目录后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
python3 /tmp/repro.py
```

不要在模型服务容器里启动该诊断进程；本次采用独立容器并暂时停止第二芯片对照服务。
