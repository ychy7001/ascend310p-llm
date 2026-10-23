# C8 K/V分核候选

基于已测device-length/transdata/compact scale/tile64，8核分为K/V各4核，替代每核在K/V之间切换。独立构建、芯片1运行，不覆盖服务库。主源码撤回无收益选项，candidate.patch保留研究实现；复现需独立源码副本应用补丁，再按build.sh指定/source编译。SHA和CMake见artifacts.txt。

三长度缓存与同C8码重建FP16及Q1 PA逐位一致；7组可变table/length Graph max_abs=0。预热5次+20次host Graph replay/synchronize计时，近期同日基线独立进程，非同进程AB。

|token|近期基线转换ms|分核转换ms|变化|
|---|---:|---:|---:|
|8192|0.313|0.306|-2.15%|
|32768|0.993|1.004|+1.09%|
|114688|3.258|3.278|+0.59%|

112K组合6.010ms vs基线5.991ms，无收益证据，小差异不作为显著退化判断。不通过性能收益门禁，不进入整模型。连同流水线和连续页候选，说明这三项调度调整未改善当前路径；不能泛化为硬件或所有转换优化不可能。下一阶段需要减少读写量/融合直接C8 Attention，同时保留精度、有效长度、页重用和MTP Graph门禁。当前112K常驻C8仍转完整FP16 NZ scratch，优化空间来自数据流而非仅页遍历。
