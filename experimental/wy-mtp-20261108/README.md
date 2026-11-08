# a2 WY / MTP 集成诊断快照

仅供实验，不是部署默认。结果和限制见 `docs/qwen36/wy-mtp-integration-20261108.md`。

这组脚本固定使用 a2 的 `/home/ychy/ascend-lab-runtime/hyperqwen-20261103` 目录、18126 端口和 `/tmp/a310-npu.lock`；本机不能直接执行设备脚本。需要既有 production manifest、只读 W8/W4 模型目录、PDF smoke 输入、`wy-boundary-20261106/tools` 中的启动与分阶段测量工具。将本目录内容按原结构复制到该实验目录后运行。

- `run_wy_mtp_matrix_20261108.py`：TP2 锁定候选；通过 `--arms` 指定 MTP 深度，`--eager --smoke-only` 是最小图兼容性隔离；不改变原镜像。
- `run_w4_wy_mtp_matrix_20261108.py`：现有 TP1 镜像；固定 native2 / torch2 / native0 / torch0 顺序。每臂 4K/8K/16K/32K 冷输入三次、4K/128 decode 三次、四页 PDF 各两次。每次分阶段测试另有一次预热。当前脚本是v2续跑快照：第一臂登记SHA并复用v1中已完成的文本结果，只续跑PDF；因此需要保留v1结果目录。v1原始脚本见本地提交 `431630c`。
- `run_pr17640_pdf_ab_20261107.py`：此前已使用的 PDF 客户端与清场快照辅助模块，SHA `0ac17615ea90295231d125b5af126d969fb5268dafab99dccfb6b15d9ed3bfc2`；本轮只复用函数，不运行其原始 main。
- `wy-mtp-20261108/*original*`：从既有 TP1 镜像只读导出的入口和 adapter。
- `*matrix*`：实验启动脚本仅支持省略 MTP 参数；实验 adapter 仅选择原 wrapper 内既有的 Torch/native WY 函数，并记录 native eligibility。native 不可用会报错；不修改 kernel 或求解公式。

每个窗口要求设备和服务空闲；用本地文件锁防重入。容器停机后保留日志和 inspect，删除实验容器，再用独立快照检查清场。失败停止矩阵，原失败不以其他执行模式通过而覆盖。

TP2 原 Graph + MTP1 实验用的是脚本加入 eager 参数前的版本；复现条件以其归档 `manifest.json` 为准。归档的初始 source SHA 与当前扩展参数后的脚本不同。

## 交付收尾诊断

W4矩阵已按用户范围调整取消，不继续运行。

- `run_w8_direct_graph_20261108.py`：W8 mode 0、capture [1,2,3]、MTP2，复用已有direct_graph_310。
- `run_w8_draft_eager_20261108.py`：在上述配置中仅将speculative-config.enforce_eager设为true，保留主模型Graph。
- `w8-bootstrap/`：复制既有direct_graph_310钩子，sitecustomize加载失败会显式终止；不包含W4权重或算子补丁。复制到a2实验根目录下`w8-delivery-bootstrap-20261108`，只在诊断容器挂载。
- 两次均在主模型SplitFuse图捕获失败、未进入请求测试，清场通过；bootstrap未封装进发布镜像。
- `verify_registry_delivery_20261108.py`：只读验证本轮两镜像的registry原始manifest和config；从任务临时auth.json读取凭据，输出不含凭据。运行完成后已删除临时auth.json；重新运行需提供registry读取凭据。该脚本不推送镜像、不改tag。
