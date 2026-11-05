# 4b-②：30 个 QKVZ 投影 W8A8

状态：生产 FP16 源量化已准备完成，性能筛选正在共享 NPU 锁队列等待。基点 `74f7202`；仅量化 30 个 `linear_attn.in_proj_qkvz`，没有叠加 MoE tiling 或共享专家量化。

量化规则已写死：每输出通道从生产 FP16 权重计算 absmax/127、nearest-even。**仅整行全零的通道使用 int8 全 0、scale=1.0**，精确表示生产 FP16 的零输出；非零通道不加 scale 下限。BF16 源仅用于记录生产转换事实，不参与量化 scale。动态激活量化使用逐 token absmax；算子筛选使用固定 seed 的合成 Gaussian 激活，不使用业务文本，无校准集调参。

全部 30 层、双 rank 共 **60** 份独立 tensor 文件已生成在原模型目录之外。Q/K/V/Z 各自 TP2 切片后，每 rank 权重为 `[6144,2048]`。每份记录 FP16、int8、scale 的内容 SHA256，汇总 manifest SHA 固定为 `ef9ae990c9af5a042787fa10f9600c538dcec2acf2e8a876bf2a40d589cac2a7`。

**生产转换观测：30 层共有 440 个通道从 BF16 非零整行转换为 FP16 全零整行，涉及 7 层；这些通道在 BF16 下的 absmax 最大值为 `6.810296326875687e-09`。** 这是生产转换事实，本实验不处理该转换。FP16 全零通道总数同为 440，rank0=270、rank1=170；所有 60 份文件均通过零通道 int8=0、scale=1.0 的精确检查。逐层逐 rank 的计数和下溢观测完整保存在 `adjudication.json`。

| 层 | rank0 全零通道 | rank1 全零通道 |
|---|---:|---:|
| 0 | 22 | 90 |
| 1 | 202 | 2 |
| 2 | 34 | 78 |
| 9 | 2 | 0 |
| 10 | 4 | 0 |
| 21 | 2 | 0 |
| 22 | 4 | 0 |

其余 23 层两 rank 全零通道均为 0。第一版准备工具遇零 scale 时拒绝继续；上层明确零通道精确表示规则后，本版继续，原始拒绝日志仍保留。

性能筛选覆盖 30 个不同真实权重，B1/B2 使用 Graph，B3/B4 eager；双 rank 顺序测量，各档3轮×256步，顺序 AB/BA/AB。B 包含每层动态量化、布局所需调用及发射成本，权重 NZ 转换在加载时完成，与 A 权重转换口径相同。先性能筛选，达到预算再进入 G0、同轮 G1/G2；没有用算子测试替代整模型门禁。

预算使用设计稿 v3 冻结的逐 batch/rank 池权重，按各档实测速比计算 `sum(pool_i*(1-1/speedup_i))`，不直接平均倍数。原始池占回放墙钟 **7.5247403423%**，理想最低倍数 **1.6630214715×**；设计稿取整 7.52% 对应 1.6637168142×。这是筛选预算，不是端到端性能结论。

| 验证 | 命令入口 | 结果 |
|---|---|---|
| CPU 权重准备 | `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 benchmarks/ascend310p/bench_qkvz_w8a8.py prepare --model MODEL --weights NEW_DIRECTORY` | 成功；60 份文件；零通道精确检查60/60 |
| 语法 | `python3 -m py_compile benchmarks/ascend310p/bench_qkvz_w8a8.py tools/summarize_qkvz_screen.py` | 2 个工具通过 |
| 预算边界检查 | 导入 `summarize_qkvz_screen`，使用真实 manifest/预算、合成计时 | 6/6：1.66×停止、2×进入门禁、2×预算值、1×零收益、0.5×负收益、拒绝不完整记录 |
| 性能筛选 | `bench_qkvz_w8a8.py bench --weights WEIGHTS --output OUTPUT --rank RANK` | 排队，未取得结果 |
| G0 / G1 / G2 | 未执行 | G1 阈值未冻结；模型 B 0次 |

| 负载画像 | 本轮收益 |
|---|---|
| 会议回放 | 未测；设计稿 2× 情景约3.76% |
| RAG 8K–32K Prefill | 未测 |
| 通用 1K–32K × C1/C4 | 未测 |
| WY Prefill | 未测；未改 WY |

真实限制：当前只有权重准备和预算工具验证；没有长上下文真实激活、饱和统计或 SSM 漂移证据，没有正式5对 AB/BA回放。预备算子筛选的双 rank 顺序运行，不含 TP2 通信，合成输入不表示上下文分布。原始 manifest、日志、量化产物位置和窗口控制记录均保留在忽略区。
