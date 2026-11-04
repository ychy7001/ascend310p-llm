# CANN9.2.1完整官方栈隔离验证（2026-11-04）

用户已同意a2隔离升级，未安装宿主合一包、未更改驱动/固件、未动a1。

官方v0.30.0-310p镜像已拉取：imageID `d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a`。实际CANN Toolkit/OPP/NNAL均9.2.1，torch2.10.0+cpu、torch-npu2.10.0.post7、vLLM0.30.0+empty、Ascend0.30.0、transformers5.14.1，无Triton。vLLM提交ced6857a，Ascend提交c09d3e20，详见runtime-inventory.json及image-inspect.json。宿主驱动26.1.1、固件7.5.0.5.220，设备Compatibility=OK。

首轮算子门禁：operator-screen-v3，默认路径无实验WY/H/O。两卡各48次输出/末state CPU容差与逐位重复性全部通过，共96/96。Dense全零activation/int8-NZ权重合成种子310..314×allocatorFalse/True共10/10PASS，采样误差0。此前旧栈同条件有间歇507015，本轮未重现；有限用例通过不能证明彻底修复。

原始失败尝试保留：operator-screen在版本守卫误拒0.30.0+empty后退出；v2因控制器漏建cache导致Podman125。两次均未执行NPU算子。v3修正版本后缀比较及提前创建cache，未改变数值容差或复现算法。

最终结论：**not_accepted_gdn_correctness**，完整默认栈尚不能采用。a3b-quality-v1的15个独立CPU参考形状/边界检查仅13/15通过，T512/decay1和8出现输出偏差；门禁阻止加载模型。随后独立新进程复核T512、seed318：HV16仅29/48在容差内，HV24为0/48；输出最大绝对误差0.00504/0.00599、最大相对RMSE0.0652/0.0749，而两组末state均在容差内且逐位稳定48/48。各组输出只有第一次与首个输出相同，提示相同输入的后续输出不稳定。

本轮未加载35B或Dense模型、未运行模型质量或性能，不宣称9.2.1修复全部问题；Dense合成10/10仅是量化Matmul环节的积极信号。完整升级同时改变vLLM/Ascend/torch-npu/CANN/transformers，不能单独归因CANN。Python chunk层仅算子名变更；原生目录重命名且多文件blob变化，不能据包装层差异推断原生算法等价。

无权重公开反馈材料已整理为results/upstream-gdn-cann921-repro-20261104.tar.gz（未发布），包含与本机测试逐字节一致的复现脚本及合成结果。最终判定见adjudication.json和gdn-focused-v1/adjudication.json。a2原服务已由控制器恢复，独立健康检查见service-restoration.json。

收尾：a2原服务独立检查active及HTTP200，无存活rootless实验容器；驱动仍26.1.1。新增Python脚本编译、启动脚本bash-n、gitdiff--check及公开复现包各文件SHA核验通过。
