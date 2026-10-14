# OrangePi ARM64 / 310P1 部署验证

环境：openEuler 22.03 LTS SP3，16 核 ARM64，系统/NPU 均报告约 87.5 GiB 总内存；驱动 7.6.T7.0.B053，固件 7.6.7.0.b054。板载 RC 设备节点使用 `/dev/svm0`。原生 NNRT 为 CANN 8.0；已有 Docker 镜像为 `local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix`，CANN 9.1、torch 2.10、torch-npu 2.10.0.post4。使用 Docker 是为了复用该机器现存 ARM64 镜像，避免重复下载占用有限根盘。

## 已验证

- 原生 ACL 初始化、设备计数、1 MiB NPU 分配和释放通过。
- 普通容器首次 PyTorch 探测报 E30006 / 507033；另有 ACL 初始化超时。共享主机 PID/IPC、挂载 `/var/driver`、`/var/dmp` 和设备管理配置后，CANN 9.1 ACL 及 PyTorch 32×32 矩阵计算通过。此处使用 privileged 作为部署探测配置，尚未收紧到最小权限。
- W4 MatMul、Pack、SwiGLU Pack 自定义 OPP 和独立 Torch 扩展在 ARM64 上编译成功。
- 使用真实 layer-0 expert0..7 权重，Gate/Up 与 Down 在 129/257/1024 行、均衡/单专家路由共 12 组单矩阵检查全部通过，误差阈值 rtol/atol 0.002。原始结果见 `w4-grouped.jsonl`；它不含模型端到端精度、路由/Attention 或服务性能。
- ModelScope 直接下载完成；52 个索引分片齐全，共 24,604,826,824 字节。索引 SHA256 与 a2 一致：`3f38112c8fc6e4f247e3c05b91df43e487e94ac7e8cf7787a01728cc87d47624`。

## 初始服务

机器目录 `/data/qwen36-deploy`，模型目录 `/data/models/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P`。
运行 `bash /data/qwen36-deploy/run-service.sh`，查看 `docker logs -f qwen36-orangepi`。
候选参数：TP1、FP16 KV、无 MTP、eager、4096 上下文、2048 token prefill chunk、单并发、内存利用率 0.45、显式 KV 1 GiB、保留视觉、开启前缀缓存、端口 18024。此文件为本机部署探测记录，尚非正式产品档。

`VLLM_ASCEND_FORCE_RC=1` 激活已有 rcfix 镜像的 RC 识别覆盖；没有该变量时可能因容器缺少 lspci 错选 EP 内存画像。`rc_kv_budget.py` 加载于本机 bootstrap 的 sitecustomize 末尾，为当前 310P worker 补上显式 KV 字节预算。另将 NPU `nn.init.normal_` 初始化移至 CPU：本机旧 AI CPU 的 StatelessRandomNormalV2 初始化曾报 0x2a/507018；不修改 checkpoint 或推理矩阵精度。

镜像 ID：`sha256:a9a8d356f62c7bbdf29fd7a44000d2bc930f124d4a48478e086ddb32a1e4c351`。
ARM64 扩展 SHA256：`e90b76dc8d3bc72b40e86d52a9fa158a911088e83c23665364ecadb35289e1af`。

## 共享内存与 48G 判断

模型加载期间观测：文件 Cached 约 26.3 GiB、服务相关进程 PSS 约 6.8 GiB、HugeTLB 约 23.2 GiB。RSS 包含共享页不能直接相加；Cached 通常可回收；大页应与设备分配记录对照，不能直接称为全部有效权重。NPU 可用内存检查不等同于 Linux MemAvailable：启动时前者只剩 51.31 GiB，而 utilization=0.75 要求 65.63 GiB，导致 vLLM 拒绝启动。

a2 的 W4+C8+MTP2+Graph 日志参考：权重 27.29 GiB、峰值激活 2.75 GiB、非 Torch 内存 0.19 GiB、Graph 0.30 GiB、分配 KV 4.30 GiB。a2 是 PCIe 独立设备，不能把这些值直接当作 RC 总内存预算；RC 还需 CPU/系统开销、加载临时量和工作区余量。

48G RC 值得验证，但 112K+MTP2+C8+Graph 尚未在限制为 48G 的同型硬件上验证。应先取得常驻/启动/长请求峰值，再确定规格。降低 utilization 只控制设备预算，不等同于验证总物理内存限制；页缓存回收也不代表权重可省去。不能声称当前 87.5 GiB 机器成功就证明 48G 可产品化。

视觉 checkpoint 张量 893,142,496 字节（0.832 GiB），MTP 张量 1,689,281,536 字节（1.573 GiB）；它们是存储量，不是完整运行时开销。保留视觉意味着还需计入图像激活与 encoder cache，前缀缓存只在命中时减少计算，不能免除未命中请求的 KV/峰值预算。

## 默认视觉与前缀缓存（用户确认）

视觉编码器默认保留，prefix cache 默认开启。视觉张量约占 checkpoint 3.63%；真实图片激活、embedding cache 和预处理缓存需另测。当前镜像源码 `MultiModalConfig.mm_processor_cache_gb=4`，LRU 缓存按 API / engine 进程复制，单 API + 单 engine 的理论总上限为 8 GiB；不是启动时固定占用 8 GiB。未来 48G RC 档应根据图片负载收紧该预算，并保留视觉功能。

开启 prefix cache 时，本机模型的 Mamba align 模式选择 block_size=1152，`max_num_batched_tokens=512` 因小于块尺寸而启动检查失败。已调整为 2048，失败日志见 `startup-prefix-align.log`。尚需实测冷/热请求、A/B/A 后缀状态复用和图片生成后才能称为默认档通过。前缀缓存命中只降低重复前缀计算，冷 Prefill 性能应继续单独报告。

本机隔离补丁复制到 `/data/qwen36-deploy/repo/bootstrap/rc_kv_budget.py`，并在同目录 `sitecustomize.py` 末尾追加一次 `import rc_kv_budget`。它不应直接移入通用官方主线。

实际服务曾完成启动和 `/health` 200，但首个文本请求触发 AI CPU `FloorDiv` 0x2a，引擎退出。定位到前缀缓存 align 的 `mamba_get_block_table_tensor` 块索引运算；隔离补丁将该小整数索引计算移到 CPU，表 gather 留在 NPU。该补丁增加一次 NPU→CPU 同步，尚需真实生成和热前缀收益测试。原始日志 `startup-first-request-floordiv.log`。此为当时的失败状态，后续基础生成通过见下一节。

### 最新峰值采样

`memory.jsonl` 中 12:11:39–12:11:59 UTC（上一轮启动/多模态预热区间）观测 HugeTLB 47.99 GiB、AnonPages 12.69–12.81 GiB、Cached 23.77–23.98 GiB、MemAvailable 最低 24.82 GiB。整机 87.51 GiB，总量减 MemAvailable 约 62.69 GiB 是粗略不可用预算指标，并非精确的模型有效内存。它包含设备大页池与 CPU 进程、系统等；不能把整个峰值归因于视觉，也不能按 checkpoint 大小判定 48G 够用。当前默认最大图片 profiling/缓存参数需要收紧并复测，才可支持 48G RC 结论。

## 最终基础生成验证（12:25–12:26 UTC）

隔离块索引补丁后，服务 `/health` 200，端口 18024。实际运行参数仍为 TP1 / W4 / FP16 KV / 无 MTP / eager / 4096 上下文 / 2048 prefill chunk / prefix on / 视觉 on；不等同于 a2 的 112K+C8+MTP2+Graph。

- 文本 `What is 2 + 2?` 返回 `4`，请求 1.956s。
- 2304 token 相同前缀、32 token 输出：冷请求 12.519s，热请求 9.812s，prefix hit counter 增加 1152；输出逐字一致。单次冷/热结果，无置信区间，整请求时间含 Decode，不能当作纯 Prefill 吞吐。
- 合成 128×128 红色 PNG 返回 `Red`，请求 1.566s。只证明基本视觉输入和生成路径可用，不代表完整视觉质量或高分辨率稳定性。

原始输出 `service-smoke.jsonl`，成功日志 `startup-success.log`。复现探测：`python3 /data/qwen36-deploy/smoke_service.py`（输出追加模式，先运行服务）。下一步是限制默认最大图片画像/预处理缓存预算并重测 RC 常驻与峰值，随后再分阶段扩大上下文和引入 a2 已验证组合。

## 完整组合验证（续）

基础 C16/MTP0/eager 的八项 toy 质量检查 8/8，异内容请求穿插后 A/B/A 文本一致、无干扰标记。真实 Chat 模板关闭思考后的共享文档 A/B/A 与独立 cold salt 共 5/5 正确；原 raw completion 检查因 32-token 思考截断不能作答案失败证据。

C16/MTP0/eager、2304/128、warmup1+repeat3、每轮独立 cache salt、零 cache hit：Prefill 中位 387.959 tok/s、Decode 中位 4.736 tok/s。当前板为 310P1/ARM64/旧固件，不能直接与 a2 的 310P3+MTP2+Graph 相比。

ARM64 C8 原始库按 Ascend310P1 编译，SHA256 `be756591b503c41a4ac0cbcc3fc515879cb56c26948de3a3d45f2d144c4c1ccb`。设备写入/追加/页读取及 GQA、因果 MTP2/MTP3 参考测试通过；KV max-abs 0.01709/0.01535。C8→NZ+原生 PA 在动态长度 1/63/64/65/127/193/256、可变页表、poison scratch 的 Graph replay 全部 max_abs=0。Graph 探针随机输入仅在 CPU 生成，真实转换/PA/capture/replay 均运行于 NPU。

新版启动器可调 `MTP=0|1|2 C8=0|1 GRAPH=0|1 MAX_MODEL_LEN KV_BYTES`；默认视觉和 prefix 开启。图片限制 1 张、1,048,576 pixels，视频暂关；预处理缓存 0.5 GiB/份。该图片边界是部署探测预算，并非图像质量或最终产品规格。

### MTP1 首轮门禁失败

C16/MTP1/eager/prefix-on/受控视觉，8题仅4/8；`703→7033`、`384→3844`、`763→7633`、`2,7,9→2,79`。因此该轮 2304/128 三次 Prefill 中位379.208、Decode5.785 tok/s不属于通过正确性门禁的有效加速结果。真实权重已加载、图片返回Red；原2304token重复输入未命中prefix（计数0），不能称为热缓存收益。下一轮关闭async scheduling作隔离诊断，尚未证明原因。

受控视觉+MTP1首轮内存采样（65样本，含停止服务）：HugeTLB峰值33.412GiB，AnonPages峰值12.534GiB，MemAvailable最低40.029GiB。相较上一轮默认最大图片的HugeTLB47.99GiB明显下降，但不同轮次也改变了MTP/KV预算，不能把全部变化归为单个视觉参数的独立收益。总物理量87.51GiB减最低available约47.48GiB，48G RC仍处边缘，且这只是8K、未包含长上下文/Graph组合；不做48G通过声明。

### MTP1 同步调度门禁通过

仅关闭 async scheduling 后，8题恢复8/8、A/B/A无串扰。2304/128三轮 Prefill中位401.672、Decode6.533 tok/s（构建进程当时在CPU运行，不能称为完全独占主机性能）；不能单独归因于MTP或同步调度。6144-token检索冷/热均精确输出ASCEND732941；冷16.510s、热4.960s，热命中4608tokens，冷prefill phase15.298s，热3.832s。表明该板的当前异步MTP组合不合格，同步调度是已测兼容路线，未证明底层根因或随机采样全面正确。

### C8 / Graph 后续门禁（进行中）

优化 ARM64 C8 库 SHA256 `939ac05a9978f8f99bdd0e4ca6991751ddd64e448c5398113741bb26556e71e4`，使用 transdata、tile64 与 compact scale，7 组动态长度/页表 Graph replay 全部 max_abs=0。模型服务采用显式 `--block-size 64`（hybrid scheduler 可再扩展页尺寸）、关闭异步调度、TE_PARALLEL_COMPILER=2；首次省略显式块尺寸的预加载已停止，未作为真实生成结果。C8/MTP1/32K/eager 模型级门禁仍在执行。GDN 三个目标 Kernel 已编译，完整包/绑定尚未验证；不能把算子探针通过写成完整模型 Graph 通过。

### C8/MTP1/eager 模型级通过

8/8 toy 正确、A/B/A一致且无串扰，小图像Red；2304/128三轮冷请求中位Prefill396.14、Decode6.576 tok/s，warmup1+repeat3且cache hit=0。6144及32000-token中点检索冷/热全部正确：32K冷88.363s（prefill87.130s）、热7.163s（prefill5.931s，hit29952tokens）。以上是单 canary，不代表正式长文质量集。

142个内存样本（含预加载重启和停止）：HugeTLB峰值33.506GiB、AnonPages峰值10.571GiB、最低MemAvailable42.151GiB；整机粗略不可用预算约45.36GiB。此轮仅32K/eager，且该值是采样包络而非48GiB物理限制验证。

GDN完整ARM64扩展SHA256 `24910050c311cb99ab5153e2b557b0e2dbe1567b1ddaec6a116c7d4d2f22fea3`。真实head形状16/32×128、非零初始状态、111/128/512/1024 tokens全通过：cosine均>.9999998，五次逐位重复一致，warmup5/repeat20。全GDN调用中位1.25–2.87ms；仅把WY换为Torch参考后约19.47–22.99ms，局部8.0–15.6倍，不是模型加速比。部署尚未开启GDN，避免混淆Graph对照。

### C8/MTP1/Decode Graph / 112K（通过首轮）

保持视觉与prefix，异步关闭，最大114688、KV6GiB、batch2048、seq1，GDN暂关。8/8 toy正确、A/B/A一致无串扰、小图像Red。日志确认FULL decode capture/replay，Graph0.43GiB。2304/128冷输入三轮中位Prefill395.596、Decode23.873tok/s，零prefix hit；此前eager/C8/MTP1中位6.576，约3.63倍（两轮上下文上限/KV预算不同，不是严格单变量等预算对照）。

|真实输入tokens|冷总秒|冷Prefill tok/s|热总秒|热命中tokens|中点canary|
|---:|---:|---:|---:|---:|---|
|32000|88.135|364.99|6.523|29952|冷/热正确|
|64000|192.316|333.82|6.854|62208|冷/热正确|
|112000|379.261|295.93|6.846|110592|冷/热正确|

检索仅生成9token，不能将8/decode phase当作稳定长输出吞吐；128token长输出单独测试。热吞吐不除完整输入冒充实际计算吞吐。112000是真实完成输入长度，114688为配置上限，未作绝对最大上下文声明。

## RC / EP 瓶颈对照（新增）

当前不能仅把差距归因RC：a2为310P3/x86 i7-14700/较新驱动，OrangePi为310P1/ARM/旧驱动；a2启用GDN、MindIEQ Prefill和MTP2，OrangePi当时原GDN、MTP1。CANN/torch大版本相同也不能消除算子/主机/固件差异。

同一 `bench-rc-ep-primitives.py`，FP16数值检查通过、warmup5/repeat30、真实NPU copy/MatMul。EP在空闲服务容器另起探针进程，RC停止服务后执行；两者内存占用背景不同。copy有效带宽按读+写字节计算，不是硬件带宽计数器。Event bracket含队列/依赖开销，不能称纯Kernel计时。

|探针|EP|RC|意义|
|---|---:|---:|---|
|128MiB copy eager有效GB/s|168.96|161.92|RC约低4%，不支持带宽数倍下降解释|
|FP16 1024×2048×1024 Graph event ms|0.16531|0.16534|本形状设备区间接近|
|同MatMul Graph host ms|0.26266|0.37993|RC主机区间较长|
|small add eager host ms|0.17878|0.36416|小操作路径约2倍|
|1MiB copy eager host ms|0.1761|0.3914|小搬运发射/同步开销明显|

这些结果优先支持继续调查小算子/主机与运行时开销，不能证明所有MoE/Attention硬件性能相同。Prefill进程20秒pidstat平均CPU43.95%（100%=单核）、wait另见日志；不支持仅凭ARM架构判定单核饱和。下一轮同参数只开启已测GDN，并启用profiler，测完非profile性能后再采集trace。

112000/128固定输出已测（warmup1+冷1+热1，样本不足以报P95）：冷Prefill314.03tok/s、Decode11.818tok/s；热Decode11.811tok/s、hit110592。与短上下文23.873存在明显下降；MTP接受率此样本100%，不能归因拒绝率。

### 已落地：ARM GDN Prefill 加速

保持同一112K配置、KV6GiB、batch2048、MTP1、C8、Graph、同步调度、视觉和prefix，仅开启已通过数值检查的GDN路径。2304/128 warmup1/repeat3且零prefix hit，中位Prefill **496.336** vs395.596 tok/s（+25.46%），Decode **23.821** vs23.873（-0.22%）。三轮Prefill分别496.336/496.045/496.414，8/8 toy、A/B/A无串扰、小红图正确。原始记录`bench-c8-mtp1-graph-gdn.jsonl`、`quality-c8-mtp1-graph-gdn.json`。

32000/128 warmup1/冷1/热1：冷Prefill498.719、Decode19.454 tok/s，零prefix hit；热命中29952，Decode19.488。warmup输出32token，不能与128输出阶段混算。此轮离线profiler导出曾在同一主机运行，长输入数据作为初测；短输入三轮没有这项干扰。原GDN-off的32K canary输出只有9token、提示词不同，不能用它计算严格长输入加速比。

### 设备trace与明确限制

当前CANN9.1包的`msprof --export=on`参数不兼容torch-npu分析入口，原始采集成功但自动导出失败。通过同包`msprof.py import`和`export summary`离线解析成功，无需升级驱动或改模型服务。复现脚本`export-profile.sh`；日志`profile-export-cold.log`、`profile-export-manual.log`。

修复`tools/profile_http_prefill.py`的热前缀干扰：warmup/profile分别使用独立cache salt，并断言profile请求prefix hit=0。首轮未隔离salt的Prefill trace不用于瓶颈结论。新的3072-token冷Prefill原始记录`profile-prefill-gdn-cold.jsonl`，设备CSV在`profile-prefill-cold/`，可运行`python3 summarize-profile.py profile-prefill-cold`复算。

- 冷Prefill设备任务时间窗5.882秒（profiling开销，不作为服务吞吐）；stream33 AI_CORE累计5.504秒。
- `QwenW4A8Int4MatmulV310`560次、累计3.593秒，占该stream AI_CORE累计时间约65.3%。各投影的mac_ratio中位0.033、vec_ratio中位0.587、mte2_ratio中位0.236。它们是本次计数器值，不是精确的端到端利用率。证据优先指向原W4路径中的Vector/解包与分组处理；不是只缺理论带宽。
- GDN WY/H/O累计约195/49/37ms；GDN已优化后，继续优化W4 Prefill比继续盲调GDN优先级更高。
- `EVENT_WAIT`等任务跨stream重叠，不能把全部task_time简单求和称为模型墙钟；汇总脚本明确排除这些事件。
- Decode采集虽然有31个主机步，但导出的设备任务只有约50.8ms时间窗，不能冒充完整64-token Decode占比。其MatMul/C8转换记录只作为路径存在证据；需补齐Graph replay采集后才能给出完整设备瓶颈排序。

RC/EP现有服务仍不完全等参：a2为MTP2、batch4096、MindIEQ shared Gate长Prefill、prefix off；本轮RC为MTP1、batch2048、原W4长Prefill、prefix on。因此不能将两端现有端到端比值称为RC模式固有损失。ARM MindIEQ原生布局算子开始隔离构建，先数值与单算子门禁，再决定是否接入模型。

### MTP2失效与MindIEQ单算子门禁

MTP2+C8+Graph+GDN、prefix/视觉保留、同步调度、114688上限/KV6GiB能够加载和capture、health200，但第一条`2+2`请求只累计一个生成token后停顿。14:52:15UTC设备TS exception `0xB406000D`，stream33/task19413；调用栈在`_get_draft_token_ids_cpu → draft_token_ids_event.synchronize`等待。证据见`mtp2-hang-stack.txt`、`mtp2-ts-exception.txt`及失败启动日志。等待位置不是触发异常的Kernel证明，当前不能归因为MTP算法或C8数值错误；该组合未通过，不提供吞吐或默认推荐。停止失败服务后另起NPU探针可正常执行。

MindIEQ ARM/Ascend310P1编译产物SHA256 `f1988a042621104837d09cfa4c24477d418cdbe3e01d13ba9bd3b62f4a97ddfa`。使用固定外部CANN许可头，仅在实验机保留依赖；本仓库仅包含构建入口、自有适配和原始测试数据。

- 6组整数参考：K/N=2048/1024、512/2048，尾块含零专家、均衡1024行、单专家4096行，全部rtol/atol=.002；8次连续提交、修改设备专家边界后Graph replay通过。warmup5/repeat20，主机同步调用中位0.41–2.68ms；无逐专家外部库比较，不称相对native加速。
- 共享Gate/Up量化：真实layer0专家0..7权重，17/32/128/512输入，codes/scale与投影逐位一致、8次队列和修改输入/路由Graph全部通过。512输入下完整Gate投影中位3.568→2.920ms（本准备路线内部对照，不是原W4模型对照）。warmup5/repeat30，见`mindieq-arm-shared.jsonl`。
- 完整模型候选仍保持MTP1，仅增加MindIEQ；需要8/8、视觉、A/B/A、冷Prefill三轮和长文检索门禁后才替换可用服务。

### MindIEQ模型阶段结果

保持MTP1/GDN/C8/Graph、batch2048与KV6GiB，增加MindIEQ native-layout、custom A8、shared Gate/Up，阈值128。2304/128三轮冷Prefill分别570.125/569.249/570.324，中位570.125tok/s（对GDN单开+14.87%，对原395.596+44.12%）；Decode中位23.297（对23.821约-2.20%）。输入SHA逐一相同、cache hit=0。该输入MTP接受率从59/69变为57/70，完整回答文本不同；吞吐改善不能充当量化精度等价证明。

基础8/8、A/B/A无串扰与小红图通过。32000/128单冷请求Prefill621.640 vs498.719（+24.65%），Decode19.487 vs19.454，128输出逐字相同且两边MTP接受率100%。warmup1/冷1/热1，热命中29952，hot Decode19.553。GDN单开32K对照曾并行离线导出profiling，且样本量小，因此该长输入加速比仍需独占重复确认。

112000-token中点检索冷/热均准确返回`ASCEND732941`。冷Prefill283.527s、395.025tok/s，冷总284.318s；热命中110592，热总5.543s。同输入SHA的原GDN-off/MindIEQ-off冷Prefill295.927，因此组合优化相对该样本约+33.49%（各单次，非P95）。仅输出9token，不报告稳定长输出Decode或全面长文精度。在长请求期间几次采样停在GDN状态索引，后续采样位置推进，最终正常完成且未新增TS异常；不能把中间等待误判死锁。

内存采样102条，覆盖本候选启动和112K冷热请求：HugeTLB最大35.645GiB、AnonPages最大10.271GiB、MemAvailable最低39.779GiB；整机总87.510减最低available约47.732GiB。仍是系统粗略预算而非模型独占量，且物理48G限制没有实测，余量偏紧，不能宣布48G RC已通过。

复现当前候选（先停止同名服务；变量只影响独立实验容器）：

```bash
MTP=1 C8=1 GRAPH=1 GDN=1 MINDIEQ=1 \
MAX_MODEL_LEN=114688 KV_BYTES=6442450944 \
bash /data/qwen36-deploy/run-service.sh
bash /data/qwen36-deploy/validate-stage.sh c8-mtp1-graph-gdn-mindieq
python3 /data/qwen36-deploy/bench-stage.py --label c8-mtp1-graph-gdn-mindieq \
  --tokens 32000 --repeats 1 --output 128 --spec
```

测试输出使用独占新文件名，重复运行需更换label。关闭MindIEQ只需`MINDIEQ=0`，回到已通过的GDN/MTP1路线。上述启动变量与路径属于本机部署探针，尚未纳入通用产品profile。

112K之后再次执行`check_cache_request_reuse.py`：8/8、A/B/A一致、无canary泄露，health200。当前容器`qwen36-orangepi`/18024保持MTP1+C8+Graph+GDN+MindIEQ、视觉与prefix开启、max114688。配置恢复时以本机`run-service.sh`显式参数为准。

## RC/EP主要配置对齐的三轮对照

临时停止a2的`qwen36-a2.service`，使用保留的CreateCommand生成独立`qwen36-ep-rc-pair`/18030；实验结束停止候选并重新start原服务单元。chip1不动。代码snapshot为本轮公开仓库，主要参数统一TP1/MTP1/batch2048/Graph/GDN/MindIEQ/seq1、prefix on、OMP4、视觉max_pixels1048576/cache0.5GiB；CLI显式KV6GiB；EP加载日志实际为6.01GiB（worker halved for workspace），GPU KV cache size=260,941 tokens，故两端预算近似但并非逐字节一致。RC使用transdata C8，EP仍原C8；不同SoC/主机/驱动/编译产物不可能消除，因此是主要配置对齐的系统对照，不是只改变RC/EP的因果实验。

同脚本warmup1/repeat3、独立salt、每请求usage/计数器有效且cache hit=0，三轮输入SHA逐一匹配：

|输入/输出|RC Prefill|EP Prefill|RC Decode|EP Decode|
|---|---:|---:|---:|---:|
|2304/128|570.290|581.511|23.396|30.111|
|8192/128|563.925|570.281|23.874|30.024|
|32000/128|514.875|506.500|19.440|22.586|

单位tok/s，中位数。8K/32K输出每轮逐字相同，2304为2/3相同，基础门禁两端8/8/A-B-A通过。8K/32K MTP接受率均100%，不能用拒绝率解释这两档Decode差距。Prefill差距在约2%内，Decode RC仍低约14%–22%。两端不同C8库意味着残余Decode差异仍混有实现差异，不能声称RC天生低此比例。

原始结果`rc-ep-pair/{rc,ep}/`，复算`python3 summarize-rc-ep-pair.py`。32K本轮提示词与早前`bench-stage.py`不同，621.640与514.875不能当作服务性能回退：此前是另一组输入，本轮两端严格逐条SHA对齐。

下一项隔离诊断为RC状态块索引：旧补丁`seq_lens.cpu()`每步产生D2H同步。设备FP32非负除法转INT32替代仅在显式开关`RC_DEVICE_BLOCK_INDICES=1`使用，64/576/1152/1728块尺寸、长度0..114688全部与CPU整数参考一致，且修改输入后的Graph replay通过；不扩展到未经验证的块尺寸/长度。默认仍为原CPU路线，模型性能/质量门禁通过前不宣称收益。

该候选模型8/8/视觉/A-B-A通过；8192/128三轮中位Prefill563.156 vs563.925、Decode23.963 vs23.874（+0.37%），3/3输出全文一致，MTP100%且cache hit=0。收益很小，不作为默认性能优化，后续已关闭`RC_DEVICE_BLOCK_INDICES`。保留仅用于诊断，不能解释RC/EP约20%的Decode差距。下一项是`MTP_DRAFT_EAGER=1`：主模型Graph、MTP2 draft eager，隔离此前draft Graph组合失败；尚未验证。

### MTP2 draft-eager隔离结果：仍失败

日志确认`num_speculative_tokens=2,enforce_eager=true`，主模型保持Graph。15:52:50UTC首个请求再次设备TS exception；device debug明确`aicpu task exception`，stream2/task21711、magic0xac79F，Python仍等待draft回传event。不能归为只由draft Graph导致，也不能将等待函数当触发算子。AI CPU日志尚未暴露具体op，仍需更细task/op关联；证据见`mtp2-draft-eager-{stack,ts,device-debug}.txt`、压缩启动日志。已停止失败候选，恢复MTP1+Graph+GDN+MindIEQ，索引候选关闭；恢复后需health/生成门禁。


### RC Decode 主机参与度补测

恢复后的 RC/MTP1/C8/Graph/GDN/MindIEQ 服务进行了 8K 输入、256 输出、三轮独占测量。服务端 Decode 阶段为 24.174、24.112、24.225 tok/s（MTP draft/acceptance 均为 128/128，prefix cache hit=0）；HTTP 从请求开始计算的总输出吞吐约 20.4 tok/s，因为它包含 TTFT。同期 `pidstat` 采集 EngineCore 平均 CPU 约 50.91%（usr 45.67%、system 5.24%），磁盘 I/O 为 0。

这说明 Decode 每步确实有明显的 host/runtime 参与，但 CPU 使用率不能直接等价于同步等待比例；它也可能包含 tokenizer、请求循环和 NPU 提交。结合 RC/EP 大矩阵和大块拷贝接近、而小操作测试差异明显，当前优先级是完整 Graph replay trace 中定位小张量更新、AI CPU task、stream/event wait 与 W4 小 GEMM 的墙钟间隔。该补测不能单独证明 RC 的 14%--22% 差距来自某一个 host 函数。原始数据：`rc-stage-8192-o256.jsonl`、`rc-decode-8192-o256.jsonl`、`rc-decode-pidstat.log`。

### MTP0 对照的配置限制

尝试在完全相同的 RC/C8/Graph/GDN/MindIEQ/114K/KV6GiB 配置下关闭 MTP（MTP0）时，模型加载和权重读取完成，但 Decode Graph capture 初始化失败：`device context length requires selected pages and one contiguous INT32 value`。因此没有产生 MTP0 性能数据，不能把该次失败解释成 MTP0 的速度问题。它说明当前实验镜像的 C8 device-length/Graph 路径对无 MTP 的上下文元数据仍有额外约束；MTP1 是当前已通过的可用档。

MTP2源码候选位置：`vllm_ascend/spec_decode/llm_base_proposer.py::attn_update_stack_num_spec_norm`（本机1590–1668附近），第二个draft新增`clamped_positions // block_size`与`% block_size`。本机另一个prefix索引路径已因旧AI CPU FloorDiv做CPU兼容补丁，而MTP2此处没有同样处理。它是待隔离验证的候选，不是已证明的异常算子，不能据此宣称修复完成。

### RC async scheduling 正确性门禁

保持 TP1/C8/MTP1/Decode Graph/GDN/MindIEQ、batch2048、114688 上限及 KV6GiB，仅开启 async scheduling。服务正常加载，health 与 smoke 通过，但固定答案检查仅 `strict_matches=4/8`；加减等请求出现错误。原始检查见 `rc-async-quality.json`。由于正确性门禁失败，没有执行吞吐测试，不提供 async 加速比。

此结果与早先 C16/MTP1 的 async 4/8 一致，说明问题不限于本次 C8 配置；仍不能证明底层触发点。已停止并保留失败候选 `qwen36-orangepi-async-ab`，恢复原同步容器 `qwen36-orangepi`。当前默认继续显式关闭 async scheduling。该实验排除了一个可直接启用的调度优化，后续优先定位 LM-head/MTP 跨流事件与 C8 小任务提交，不能把设备任务累计占比直接当作可消除的墙钟时间。

同步恢复后文本、图片 smoke 成功，固定答案重新通过8/8，A/B/A文本一致、无canary泄露，见 `rc-sync-restored-quality.json`。本轮没有得到可发布的 Decode 加速；服务仍保持原同步基线。

### TASK_QUEUE_ENABLE=2 兼容性门禁

在完全相同的 RC/C8/MTP1/Graph/GDN/MindIEQ/114K/KV6GiB 配置下，仅将 `TASK_QUEUE_ENABLE` 从默认1改为2。服务能够进入权重加载，但在 Decode Graph 捕获阶段失败，运行时明确报错：`Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture, please export TASK_QUEUE_ENABLE=1/0.` 因此没有质量或吞吐数据，不能把该开关用于当前 Graph 产品档。失败日志保留为 `startup-taskq2-failed.log`；测试容器已隔离，`qwen36-orangepi` 已恢复并通过 health、文本/图片 smoke。

这也解释了为什么没有直接尝试用 Level-2 任务队列解决 RC Decode 差距：官方调优文档虽建议 CPU 瓶颈模型尝试该变量，但当前 310P Graph 实现先限制了组合。后续若要继续，只能建立关闭 Graph 的独立对照，不能与当前产品基线直接比较。

### CPU_AFFINITY_CONF=1 对照

在同步调度、C8、MTP1、Decode Graph、GDN、MindIEQ、114K/KV6GiB 全部不变的情况下，单独开启 CANN 文档建议的 `CPU_AFFINITY_CONF=1`。候选通过固定答案 `8/8`、A/B/A 一致性和视觉 smoke；8K/256 冷 Decode 阶段为 `23.975 / 24.032 / 24.057 tok/s`，热请求为 `24.179 tok/s`。恢复基线此前同条件约为 `23.874--24.225 tok/s`，两组重叠，不能证明稳定收益。候选原始数据为 `quality-cpu-affinity1.json` 和 `bench-cpu-affinity1-8192-o256.jsonl`，容器已停止并保留为 `qwen36-orangepi-cpu-affinity1`，默认服务仍不设置该变量。

### MTP draft eager 对照

将 `MTP_DRAFT_EAGER=1`，只让 MTP draft 分支使用 eager，主模型仍保持 `FULL_DECODE_ONLY` Graph。固定答案 `8/8`、A/B/A 和视觉 smoke 通过，但 8K/256 冷 Decode 为 `22.753 / 22.762 / 22.806 tok/s`，热请求 `22.859 tok/s`；同步 Graph draft 基线约为 `23.975--24.057 tok/s`，下降约 5%。因此当前 310P 上 draft Graph 是有效优化，不能为了规避潜在的 draft 同步问题而默认 eager。结果保存为 `quality-draft-eager.json`、`bench-draft-eager-8192-o256.jsonl`，候选容器已停止并恢复 Graph draft 基线。

### C8 int32 元数据复用对照

新增显式开关 `QWEN36_C8_REUSE_INT32_METADATA=1`：只有当 slot/page table 已经是 NPU 上连续的 `int32` 时才复用原张量，否则仍走原来的转换路径。该开关不改变 C8 算子协议和 Graph shape。两轮 8K/256 冷 Decode 分别为 `24.042 / 24.148 / 24.106` 和 `24.188 / 24.189 / 24.177 tok/s`，热请求分别为 `24.157` 和 `24.164 tok/s`；固定答案8/8、A/B/A和视觉 smoke通过。相对原同步基线约 `23.87--24.22 tok/s`，收益约 0.5%--1%，属于小幅但可重复的微优化。默认仍关闭，建议在固定的 C8 Graph 版本上显式开启并继续观察。
