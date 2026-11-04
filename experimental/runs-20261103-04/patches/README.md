# FwdO 失败实验补丁

`production-fwd-o-vector-mask.unvalidated.patch` 与 `production-fwd-o-mask-fences.unvalidated.patch` 均已完成隔离构建和固定输入门禁，各 0/48，通过配对对照的原算子为 48/48。两者均被排除，禁止合入、覆盖生产 OPP 或启动模型。文件保留用于复现失败。

向量 mask 写法参考 GruntovDima PR17640 固定 head18163f6b 的短行处理；fence 试验保留原算法，仅尝试 V/S 同步。原函数从 row56 开始用 SetValue 写短尾，与 token56 偏差位置吻合只是排查线索，不是根因证明。

同构建框架下，原始 FwdO 源码只改独立 API 名称，固定输入 48/48 通过。完整链仍有调用历史相关异常。证据见 results/fwd-o-maskfix-vector-rejected-20261103、results/fwd-o-maskfix-fences-rejected-20261103 和 results/fwd-o-rebuild-control-20261103。不能把两补丁的失败简单解释为构建框架不兼容，也不能把原算法单算子通过当作完整链正确。
