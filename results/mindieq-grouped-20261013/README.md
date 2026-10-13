# MindIEQ设备端分组原型（2026-10-13）

独立Ascend C扩展，在一轮kernel launch内由设备读取int64专家累积行数，依次复用MindIEQ固定310P类的W4 biased解包、INT8 Cube、corr/scale和输出路径。8个核根据累计tile数量轮转起点；固定m0/n0/k0=64/128/256。量化输入A8 ND，权重E×packed NZ，per-token/per-channel fp32 scale，输出FP16 ND。没有每token完整权重全局反量化、没有专家counts回读CPU。实验不等于正式vLLM后端。

外部类按固定commit从已有构建树生成；保留外部CANN许可，公共仓库只包含自有wrapper和生成器，不包含生成的外部类。新代码不改变稳定OPP、权重或服务启动参数。真实服务未采用。

## 构建与故障修正

首次缺少生成头路径，继而缺少CubeFormat namespace，修正后设备编译完成但因空CMAKE_BUILD_TYPE对象合并失败；补Release编译完成。首轮NPU在tail_zero的行0/242失败，其余行有限：对应M1及M129尾部单行。原ND单行路径为V220 helper，原型改为统一32元素K分形加载后通过。输出Nz2Nd采用m_actual/n_actual，避免覆盖相邻专家。量化A与scale末尾额外16行零padding供尾部块加载，不影响返回行数。

固定两投影真实形状GateUp[K2048,N1024]、Down[K512,N2048]；合成signed INT4权重和INT8激活。输入分布tail_zero=[1,0,2,15,16,17,63,129]、balanced=8×128、one_expert=4096+7×0、三真实counts=256专家4096routes。12案例全部对CPU整数参考通过，8次连续提交输出通过、固定Graph更换设备专家边界（counts逆序，同总行数）后参考通过。精度rtol/atol=.002/.002，结果含FP16输出误差，不等同checkpoint质量。

## 单投影计时

warmup5，重复20，分组/逐专家交错，host dispatch+synchronize耗时，权重打包/输入准备不计入；逐专家用已验证caller-workspace绑定。原始20次采样见mindieq-grouped-v2.jsonl。

| 投影/真实counts | grouped ms | 逐专家MindIEQ ms |
|---|---:|---:|
| GateUp p05 |5.272|9.700|
| GateUp p50 |5.228|8.609|
| GateUp p95 |4.096|6.193|
| Down p05 |3.599|5.761|
| Down p50 |3.614|5.419|
| Down p95 |2.778|4.032|

固定tile对one_expert4096退化：GateUp1.901 vs0.698ms，Down1.506 vs0.935ms；不支持“所有shape必然更快”。p05/p50/p95为counts选样，非性能分位。不能把单投影加速直接外推模型Prefill。

## 全部256真实专家权重的完整MLP

第一轮三真实counts均通过，四后端×8次提交×3案例=96输出重复逐值一致；CPU合成FP16激活，计入量化/SwiGLU/padding分配/两次grouped调用，warmup5、20次测量。

| counts | 原W4 ms | MindIEQ逐专家 ms | MindIEQ grouped ms | W8逐专家对照 ms |
|---|---:|---:|---:|---:|
| p05 |19.368|27.601|9.836|15.615|
| p50 |19.057|24.930|9.772|14.091|
| p95 |17.343|17.158|7.825|10.055|

相对原W4耗时节省约49%/49%/55%。grouped与逐专家MindIEQ在此输入下输出与native的误差指标相同（相对L2约1.043%～1.062%，来自不同激活量化器）；不是模型质量评估。W8逐专家也含Python调用，不能声称超过最优grouped W8。完整MLP未测Graph，本轮已测Graph范围仅单投影。

独立第二轮三分布及96连续输出全部通过、退出0；grouped中位数9.870/9.796/7.839ms，native19.324/19.080/17.279ms，相对耗时减少约48.9%/48.7%/54.6%。服务集成需避免两种W4布局全部同时驻留：现有native为[n16,K64,n16,32B]signed，MindIEQ为[K64,n16,n16,32B]biased。需要片上按native布局读取或统一布局适配；直接另存全部专家会损失TP1 KV内存。不能把此isolated MLP成绩写成模型Prefill收益。

此前逐专家全量替换失败的性能门禁保持。复现需要本轮固定外部源码、镜像、CANN、权重和路径，先build-grouped.sh新目录构建，再顺序run-grouped-v2.sh与run-grouped-real-mlp.sh，芯片0独占。自定义build目录须同时替换run脚本mount路径。生成头输出必须不存在，外部原文件不改。原始编译和失败结果均保留，最新扩展hash见artifacts.json。

增量重建曾出现m200对象格式错误，使用grouped-build-clean新目录编译成功；新host绑定拒绝minimum_rows非零，避免跳过专家返回未初始化行。该参数目前只支持0。本轮不声称实现混合阈值调度。

新目录构建对应提交代码的tail_zero/balanced两投影共4例，整数参考、连续提交、动态counts Graph均通过，退出0。该host约束修改不改变设备代码；完整MLP两轮使用此前host默认minimum_rows=0。

CPU字节关系检验覆盖4形状：native按[K64,N16]转置并对字节XOR0x88，与原MindIEQ pack逐字节一致。该检验不代表设备布局转换或片上读取完成；复现工具check_mindieq_native_layout.py。本机无Torch/NumPy，最终在既有镜像的CPU模式完成，不初始化NPU。
