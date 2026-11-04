# FwdO fence 补丁失败记录

隔离 API ChunkFwdOMaskFix，生产原算法仅在标量 mask 短尾前后添加 V/S fence；固定输入候选0/48，配对生产48/48。状态failed，未进入完整GDN门禁或模型启动。

后续同框架无算法改动的独立重编译48/48通过，见results/fwd-o-rebuild-control-20261103。没有覆盖生产OPP，a1未修改。
