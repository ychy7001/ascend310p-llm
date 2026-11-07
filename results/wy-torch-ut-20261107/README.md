# WY Torch UT隔离候选（2026-11-07）

本轮面向冷Prefill优化，只在a2实验，不推送、不切生产。

## 1. blocked UT：固定真实输入精度不通过

实验镜像d103中的冻结Torch WY源码SHA为`4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f`，仍使用逐行UT。
参考源码`e024ecaee543964c44c6e80fa1c1ed5baaa1ec5e04df98c3555615bcbc19295d`已使用blocked inverse。此前将参考实现当作实验基线的判断已撤回。

仅替换UT，保留原Gram、衰减、W/U路径。单个捕获的完整WY从6.42ms降为4.85ms，只是微基准，不能折算为端到端Prefill收益。

固定132快照/触发与480边界输入全部执行。612组同步/排队重复通过，q/k/g逐位一致。但独立CPU裁决不通过：快照候选94/120、基线98/120；边界均403/480，候选W/U平方误差及U超差元素均较基线增大。详见`blocked/cpu-summary.json`。

未执行合成H/O、完整G1/G2或性能门禁，未采用。冻结阈值、协议、worker、R与生产kernel未改。

`blocked/replay-torch-ut-g0.py`与`analyze-torch-ut-g0.py`是此次诊断脚本快照，保留固定/runtime和/work挂载契约，需要冻结镜像、捕获输入及源码文件；不是可独立运行的生产实现。原始3.1GiB设备输出保留在workspace的`tasks/tmp/a310-wy-localize-20261106/torch-ut-g0-20261107`，不进入git。清场和原始证据SHA见`blocked/evidence.json`。
