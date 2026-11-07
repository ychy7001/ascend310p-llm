# 原Torch UT算术的NPU图回放探针（2026-11-07）

原Torch WY模块SHA `4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f`。仅将63次row/sub clone、乘、sum、add及最终eye放入NPUGraph，算术顺序与dtype不变；每次将新attn复制进固定输入buffer再replay。未调用native WY或修改生产模块/worker/协议/阈值/R。

先只读核对当前镜像graphs.py（SHA `f8ba6f797542c3a2ee874f4e3e764c33f7926cb536ec7c9af1160110e6c1aa5d`）提供NPUGraph/graph接口。首窗口错误地把ASCEND_CUSTOM_OPP_PATH设为空，CANN初始化读取custom op JSON失败，13.63秒，ExitCode 1，未进入捕获；独立清场通过。该窗口保持失败，不作为图能力结论。

第二个env-fixed独立窗口恢复既有Torch诊断的paired vendor路径，13.52秒，ExitCode 0、独立清场通过。冻结输入为rank0/rank1捕获各一个T128和T640，均在设备0运行。四例各3次、五项输出均与原Torch逐位一致；两形状共享缓存，rank输入切换后仍一致。原UT算术没有融合、重排或改归约。

| 捕获来源 | T | 8对AB/BA速度比中位 | 更快对数 |
| --- | --- | --- | --- |
| rank0 | 128 | 1.23050 | 8/8 |
| rank0 | 640 | 1.03020 | 8/8 |
| rank1 | 128 | 1.24463 | 8/8 |
| rank1 | 640 | 1.02866 | 8/8 |

计时含完整WY调用、每段10次、各预热5次，图仅覆盖UT；属于微观诊断，不是模型/性能采用证据。每个图的capture时间约3–6毫秒，最大时点进程allocated约35.3MB、reserved约73.4MB，不能将进程总内存当成图的增量内存。长Prefill尚未测；不能沿用native WY的37–47%吞吐收益。

下一步固定612组、双设备、串行/排队重复验证全输出字节等价，并记录实际形状和内存；身份通过后再测较长真实捕获的稳定微观收益。原Torch本身不满足此前native绝对G0，因此这里是保持基线语义的等价调度线，不能写成native绝对G0通过。尚未进入模型筛查或生产实现。
