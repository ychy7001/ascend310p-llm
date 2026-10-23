# C8 Cube宽PV候选

在已有直接C8 Cube Attention原型中，将每16个历史token、每16个输出维度的多次PV Mmad改成一次`M=16,N=dim,K=16`宽PV Mmad，期望减少循环和同步。首版多查询数值失败：L0C读回按16×16列块排列，误按行连续读取；保存于check-v2前的失败日志。修正列块索引后重新编译通过。

修正版在芯片1通过Q1/多查询、dim128/256、splits4的精度和Graph页表/查询变更检查：check-v2最大kernel误差约8.1e-5；长路径三长度与FP16 reference通过。计时为5次预热+20次launch+synchronize，含预分配输出，非纯kernel。

|路径|8K|32K|112K|
|---|---:|---:|---:|
|native FP16 PA|0.311ms|1.123ms|2.927ms|
|direct fused C8 Cube|1.722ms|7.017ms|24.229ms|
|C8转换+native PA|0.561ms|1.947ms|6.192ms|

Direct fused C8在112K约为native PA的8.3倍；因为每个Cube原型核仍需在线INT8反量化、QK/PV、online softmax及大量同步，不能以省掉NZ scratch推断会更快。staged路径也略慢于此前基线组合约6.03ms。该候选不接入服务；主源码已恢复，candidate.patch和原始日志保留用于后续彻底重写。后续若继续直接C8 Attention，应改为真正的块级Cube/softmax流水线而非当前逐小块原型。
