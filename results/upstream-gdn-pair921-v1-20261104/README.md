# CANN9.2.1配套H/O独立正确性验证




固定官方镜像ID d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a，使用同一工具链重新编译PR17640固定head18163f6b的成对H/O，独立vendor/namespace，保留原始NZ状态binding。**Torch WY保持默认**，没有加载旧9.1 WY二进制或改算法。

| 检查 | 新版默认栈 | 9.2.1配套H/O、Torch WY不变 |
|---|---|---|
| T512/seed318/HV16，device0 | 29/48在容差内，输出1/48等于首轮 | 48/48在容差内且输出/末state逐位稳定 |
| T512/seed318/HV24，device0 | 0/48在容差内，输出1/48等于首轮 | 48/48在容差内且输出/末state逐位稳定 |
| 同输入device1 | 本轮默认对照未覆盖 | HV16、HV24各48/48通过 |
| 15组严格边界 | device0为13/15 | 两卡各15/15，每组额外3次逐位重复 |

配套H/O的T512输出maxabs为6.54e−6/6.38e−6，relativeRMSE约0.000424；15组输出最大maxabs1.025e−5、末state1.2465e−4，原门限atol2e−4、rtol0.04、relativeRMSE<=0.02未改变。结果说明此配套组合能消除已测样本异常，尚未区分H或O根因，不是整模型质量/性能验收。

验证前纠正加载顺序：vLLM bootstrap会将默认vendor置前，而9.2.1默认库也导出相同Vllm API名；独立NPUBridge会在dlopen时捕获环境。故在bootstrap之后、dlopen之前重新前置paired vendor，并记录API/扩展SHA和实际映射库。控制器在未运行验证时暂停，保留旧脚本和调整记录；算法/编译源未改。6个验证进程均只映射配套libcust_opapi.so。原始通用probe头仍称default/original scope，实际实验范围以wrapper日志和adjudication为准，未修改原始jsonl。

证据：results/upstream-gdn-pair921-v1-20261104/。a2原服务已独立HTTP200恢复后，继续进入35B W8A8 TP2质量筛选，配置为9.2.1+paired H/O+原Torch WY；暂不做WY性能比较。

归档不含编译源码树、OPP二进制和扩展缓存；远端保留。operator-artifacts.json记录所有运行包文件SHA；paired-binding.cpp为实际生成绑定。source-provenance.json核验203个源码blob，tools为实际执行脚本；wrapper-before-loader-order.py为尚未运行NPU时保存的前版。service-restoration.json为该算子轮完成后、模型质量轮开始前的独立恢复采样。
