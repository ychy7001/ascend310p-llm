# CANN compute-WY 可行性核查（2026-11-04）

开始08:33 UTC，截止10:33 UTC。仅本机CPU构建；实验机仅只读已有镜像文件，不操作服务、不启动NPU容器、不访问另一实验机。其他thread新增文件保持原样。

计划：核实#18057当前head与固定ops-transformer 0340eae；检查9.2.1镜像实际算子/绑定；按该提交实际构建两个自定义算子；评估WY与gating拆分及310P支持；官方核实MR合入/发行版。结果仅提交README和adjudication，编号wy-1: 8，保留原始日志SHA。

已核实：
- GitHub官方API：#18057 open，head 2620e77d7bdb3c54c6ea98b24b570269f6eed6b7；固定ops提交0340eae051ddf5ab956df0f7e0396dbd6314cfc5。已从官方GitCode克隆并checkout。
- 官方MR API均state=merged：!11242于2026-09-17 10:47:29+08:00，!12078于同日17:20:01+08:00；官方git记录及祖先关系确认固定提交包含两者和统一绑定重构!13630。现有18个公开tag均无两个experimental目录；9.3.0-beta.1分支包含，不当作已发布版本；releases API为空，具体发行版未核实。
- 实验机仅只读镜像层：47587文本、11份opapi库均未发现两算子导出/头文件/op信息/目标Python绑定。通用cann_ops_transformer包存在，不能混同目标experimental绑定。镜像层扫描不解析whiteout，最终用本机同ID实际rootfs复核。最初扫描错误跟随镜像绝对symlink，已修正，不涉及容器/服务操作。
- #18057相对镜像基线：WY文件+57/-1、加载器新23行；gating+25；gdn入口+9/-5（其中有基线差异带入的状态写回改动）。完整PR12文件+641/-4，不能整包套用作WY单变量实验。
- vendor_name控制独立vendor；选定ops的wheel可使用独立Python包名，但Torch dispatcher namespace固定cann_ops_transformer，需明确与包名隔离的区别。
- 本机公共镜像仍在下载；CPU-only构建脚本已准备，分别记录run包和wheel的退出码，不映射NPU或驱动。

最终完成：
- 本机拉取同ID镜像完成；实际rootfs扫描78033份文本和11份opapi库，均无目标导出、头文件、opinfo及绑定；基础cann_ops_transformer存在。runtime源码SHA与已下载镜像基线c09d3e20一致。
- 约08:59–09:02 UTC完成固定0340eae的custom run包、experimental wheel、定向独立包名wheel，全部退出0。另约09:04–09:05 UTC实际完成wy921_cann vendor包，退出0。未修改上游算子源码，构建依赖仅安装在本机CPU容器内。
- 两native包提取核验通过：四ACLNN导出、两头文件、两310P opinfo和两kernel ELF；两wheel均含目标Python接口与C++源码。独立vendor及Python包名可用，Torch namespace和部分utils导入仍硬编码，严格私有namespace需薄绑定。
- 两CPU容器已自动删除；非privileged、未挂NPU/驱动，实际设备扫描为空。a2仅只读已有镜像层，无服务停启或容器启动，没有接触其他thread的运行/文件。
- 发布核查扩展到18个tag的完整文件树，未找到目标算子路径；具体发行版仍标未核实，不能把9.3 beta分支当发布证明。
- 最终结论：CPU路线可行，需要先完成WY独立薄绑定、保持原gating/H/O，再在NPU释放后做桥接/设备算子门禁并进入G1。lazy JIT桥接未编译/加载，数值与性能均未验证。
- 完成结果README/adjudication、实际制品审计、脚本语法及证据SHA检查；全轮约40分钟，按wy-1: 8本地提交，不推送。任务归档。
