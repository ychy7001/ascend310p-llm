# 上游 FwdO 单独替换组合：未通过

使用 GruntovDima PR17640 固定 head18163f6b 的 FwdO 原实现（无算法修改），生产镜像构建、独立 API/torch namespace，保留生产 FwdH。固定FwdO输入48次：上游FwdO 0/48在容差内，但48/48逐位稳定；配对生产48/48在容差内且稳定。候选maxabs约0.004327、relative RMSE约0.195875，受影响token0～6及64～70。严格门禁提前停止，未进入完整GDN门禁或模型测试。

只能说明“上游FwdO + 当前生产FwdH/输入”组合未通过；并未测试该PR完整FwdH+FwdO，不可外推为上游完整PR错误。没有覆盖生产OPP，a1未动。
