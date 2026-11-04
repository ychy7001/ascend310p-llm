# Dense post4完整上下文容量和短正确性筛选

在post4 v3同镜像/同步eager配置基础上，仅利用率0.90→0.94；最大上下文仍262144，batch4096/seqs4/prefix on/MTP off不变，未更换权重或量化方法。

两rank各19.2416GB权重，缓存预算8.50GiB（另一rank8.63GiB），报告KV274795tokens/满长请求并发1.05×。已完成缓存初始化并返回health200；这里是容量准入和短请求，不是实际256K长请求或多路满长并发测试。

quality-before8道严格匹配7/8，排序题期望2,7,9实际279；其余7项正确。3072token干扰前后NPU7329输出一致、无ZXQ9182泄漏。原始status为failed(strict checks)，未松门槛；quality-after、GSM、业务语义与正式性能基线均未执行。

该结果不能作为已通过验收或可上线配置。控制器停止自己的Dense容器并请求恢复a2临时unit；后续实验另行占用a2，最终服务状态需重新核验。
