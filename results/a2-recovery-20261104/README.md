# 原服务恢复记录（2026-11-04）

对象：目标实验机上的 `llm-backend-primary.service`。时间使用 UTC；北京时间 CST = UTC+8。

**最终结果：第二阶段已恢复。** 停止竞争服务并释放显存后，原服务仅启动一次即成功；13:14:32–13:14:33 独立 health 均为 200，唯一一条短问题返回正确答案。13:14:58 服务 active/running、NRestarts=0。

## 根因判断与证据

- 第一阶段不可用的直接原因是 NPU 显存竞争。另一项用户级服务 `qwen36-image.service` 于 11:58:28 启动，其容器 `qwen36-image-managed` 的 EngineCore（宿主 PID 3061）在 NPU 0 占用 37124 MiB。进程 cgroup 与容器状态相互印证。原服务使用 TP=2、显存利用率 0.9，每颗芯片启动检查要求约 38.92 GiB 可用。
- 本轮开机后，原服务首次加载于 12:00:27 报 NPU OOM：申请 578 MiB 时仅余 703.23 MiB；后续两次自动重启均出现显存可用量低于启动要求的错误。
- 诊断开始时服务自 12:02:06 起显示 active、NRestarts=2；但 12:02:54 已报 `EngineCore failed to start`，12:02:55 已报 `Engine core initialization failed`，之后日志停滞近半小时。容器仍 running，残留 API/辅助进程且有僵尸进程；18081 无监听，health=000（连接被拒绝）。判定为初始化失败后卡住，此前曾自动重启，非正常模型加载。
- 当时双芯片 npu-smi Health 均为 OK、AICore=0%；芯片 0 总占用 38610/44277 MiB，芯片 1 为 1417/43693 MiB。未发现要求宿主机重启的硬件健康证据。
- 主存约 31 GiB、available 23 GiB；交换区 8 GiB 已满。内存 PSI 的 some/full 在 10/60/300 秒窗口均为 0，当前服务 cgroup 的 oom/oom_kill 计数均为 0。磁盘使用率 37%、可用约 1.1 TiB，inode 使用率 3%。交换区已满需记录，但当前直接启动阻断由 NPU 显存错误证实。
- 容器内模型目录、配置与权重索引可读；索引引用的 10 个 safetensors 分片均存在，每个分片的文件头抽样读取成功，总大小 39767942304 字节。此检查未读取全部权重、未做 Profile 解析。
- 上次开机的内核日志确认发生全局 OOM：10:33:51 开始出现 oom-killer；10:35:12 至 10:52:43 多次杀死进程，包括多个 Python 进程及用户级基础进程。Python 单进程匿名常驻内存约 4.5–6.2 GiB，多个受害进程属于同一旧容器。现有记录未将该旧容器可靠映射到 Profile 解析任务，因此只确认全局 OOM，不能断言由并行解析触发。
- 用户描述约 17:40 CST（09:40 UTC）失联；已检出的 OOM 日志始于 18:33:51 CST（10:33:51 UTC），时间并不重合，不能用后者直接证明前者的起因。当前开机时间为 11:58:09 UTC（19:58:09 CST）。

## 操作与时间

| 时间（UTC） | 操作与结果 |
| --- | --- |
| 12:31:02–12:32:41 | 只读检查服务单元和日志、容器状态与进程归属、双 NPU、内存/交换/磁盘、模型可读性、监听端口及前次开机 OOM 日志。 |
| 12:32:41 | 第 1 次人工调用 `systemctl restart --no-block llm-backend-primary.service`；新容器于 12:32:51 启动。 |
| 12:33:35–12:33:57 | 显存检查失败：5.85/43.24 GiB 可用，低于要求的 38.92 GiB；12:33:43 EngineCore 失败，12:33:44 API 初始化失败，12:33:45 退出。12:33:57 观测确认失败，health=000；系统按原 Restart=always 配置于 12:33:55 自动重试。第 1 次观测在约 76 秒内确认失败。 |
| 12:34:19 | 第 2 次人工调用同一服务的 restart；新容器于 12:34:25 启动。 |
| 12:35:02–12:35:24 | 再次显存检查失败：5.90/43.24 GiB 可用，低于要求的 38.92 GiB；12:35:10 EngineCore 失败、12:35:11 API 初始化失败、12:35:12 退出。12:35:24 观测确认失败，health=000；本轮约 65 秒内确认失败。12:35:23 系统自动重试被原有启动频率限制拒绝（Start request repeated too quickly）。停止进一步恢复操作。 |
| 12:36:04 | 独立重新请求 health：000，curl 退出码 7（连接被拒绝），18081 无监听；服务 failed/failed，MainPID=0、ExecMainStatus=1。双 NPU 仍 OK，竞争容器及 PID 3061 自启动以来未变，仍占用 37124 MiB。 |

人工 restart 调用严格限于上述两次。systemd 自行触发的重试单独记录。每轮等待上限 15 分钟，出现明确的 EngineCore 初始化失败即可提前确认失败。

## 第一阶段验证（历史结果）

第一阶段的两次人工重启均确认初始化失败，当时未恢复。该阶段独立复核时间为 **2026-11-04 12:36:04 UTC（20:36:04 CST）**：

- `18081/health=000`，curl 退出码 7，连接被拒绝；18081 无监听。
- 服务 `failed/failed`、MainPID=0；第二次失败后的自动重试因原有启动频率限制被拒绝。
- **未发送最小推理请求**：服务未达到 health=200，无法验证生成。
- 最终 NRestarts=1 是最近一次人工 restart 后的自动重试计数，不代表本次人工 restart 次数。人工调用次数为 2，已用尽授权次数。

## 第一阶段处置建议（历史记录）

第一阶段建议由用户协调释放竞争服务占用的 NPU 显存。当时尚未执行以下步骤；后续授权与实际执行记录见第二阶段。

1. 协调竞争服务使用方后，在对应用户会话执行 `systemctl --user stop qwen36-image.service`，确认 `qwen36-image-managed` 退出。应从管理服务停止，避免仅停容器后被管理服务重新拉起。
2. 用 `npu-smi info` 确认 NPU 0 上 PID 3061 消失，原服务所需两颗芯片各自有至少约 38.92 GiB 可用显存。
3. 执行 `sudo systemctl reset-failed llm-backend-primary.service` 清除启动频率限制，再执行 `sudo systemctl start llm-backend-primary.service`。
4. health=200 后发送一次单条短问题，确认生成正常。

现有证据不要求宿主机重启、驱动变更或软件安装。

## 第一阶段变更范围

远端仅对指定原服务发起两次 restart；未修改单元、配置、模型或驱动，未安装软件、重启宿主机或操作其他容器。未访问另一实验机，未启动 NPU 实验负载，未运行远端 Profile 解析，未删除任何实验文件。本任务本机仅写入本脱敏摘要，未修改仓库已跟踪文件；已有并行工作保持原状。

## 第二阶段：a2fix-2 授权恢复

用户追加授权停止竞争服务，并明确允许原单元 `--rm` 在停止时自动清理竞争容器及可写层；镜像、挂载模型数据和单元文件保持原状，不额外执行删除命令。

| 时间（UTC，2026-11-04） | 操作与证据 |
| --- | --- |
| 13:08:59 | 停止前复核：竞争服务 active/running、enabled；NPU 0 为 38609/44277 MiB，NPU 1 为 1418/43693 MiB；竞争 EngineCore PID 3061 占用 37124 MiB。 |
| 13:09:00 | 仅调用一次 `systemctl --user stop --no-block qwen36-image.service`。 |
| 13:09:13 | 原单元 ExecStop 成功结束（status=0）。 |
| 13:09:23 | 独立复核服务 inactive/dead、Result=success，仍为 enabled；当时容器列表显示 Exited (0)。NPU 上无运行进程。 |
| 13:09:53–13:09:54 | 确认两颗芯片显存充足后，依次调用 `systemctl reset-failed llm-backend-primary.service`、一次 `systemctl start --no-block llm-backend-primary.service`；未再次 restart。启动等待截止时间 13:24:53 UTC。 |
| 13:10:37 | 两名 TP worker 均进入模型加载。 |

停止前后显存对比（npu-smi 的 MiB；可用 GiB = (总量−已用)/1024）：

| 芯片 | 总量 MiB | 停止前已用 MiB | 停止后已用 MiB | 停止后可用 GiB |
| --- | ---: | ---: | ---: | ---: |
| 0 | 44277 | 38609 | 1571 | 41.71 |
| 1 | 43693 | 1418 | 1394 | 41.31 |

停止后两颗芯片均超过约 38.92 GiB 的启动要求。主存 available 约 29 GiB，交换占用约 26 MiB。

## 第二阶段验收

| 时间（UTC，2026-11-04） | 验证结果 |
| --- | --- |
| 13:11:24–13:11:30 | 权重加载完成，两名 worker 各加载约 18.8544 GB。 |
| 13:12:48 | 引擎完成编译、KV cache 初始化及图捕获。 |
| 13:13:13 | API 日志显示 Application startup complete。 |
| 13:13:22 | 等待循环首次测得 health=200，距启动约 3 分 29 秒，未超过 15 分钟上限。 |
| 13:14:32.946646 | 使用新 SSH 会话独立请求 health，HTTP 200，耗时 0.002 秒。 |
| 13:14:32.946719–13:14:33.400845 | 唯一一次推理 POST 请求成功，HTTP 200，耗时 0.454 秒。 |
| 13:14:33.401450 | 推理后再次独立请求 health，HTTP 200，耗时 0.001 秒。 |
| 13:14:58 | 原服务 active/running、Result=success、NRestarts=0；18081 已监听。竞争服务 inactive/dead、Result=success、仍为 enabled；其容器仍列为 Exited (0)，未额外清理。 |

推理请求使用 `/v1/chat/completions`，模型 `Qwen3.6-35B-A3B-w8a8`，单条 user 消息为“1+1等于几？只回答数字。”，temperature=0、max_tokens=32、enable_thinking=false。实际返回 **`2`**，finish_reason=`stop`，输入 22 tokens、生成 2 tokens、总计 24 tokens。没有重发推理请求。

最终 npu-smi：双芯片 Health=OK；芯片 0 占用 38010/44277 MiB，芯片 1 占用 37786/43693 MiB，显存主要由原服务的两名 TP worker 使用。此占用是恢复后的正常服务分配，与启动前检查显存释放的阶段不同。主存 available 约 25 GiB，交换区已用约 6.7/8 GiB。

## 第二阶段操作边界与真实限制

- 远端变更仅有一次竞争服务的 systemctl stop，以及原服务的 reset-failed 和一次 start；没有新增 restart。未 disable 服务、改动单元/配置/模型/镜像/驱动，也未额外执行容器或数据删除命令。原单元的自动清理副作用已获明确授权；最终实际观察到竞争容器仍为 Exited (0)，故保留原状。
- 竞争服务仍为 enabled，手动启动它或宿主机重启后可能再次产生显存竞争。本次恢复不包含持久化消除该启动冲突。
- 已验证 health 与一次短文本生成；未进行长上下文、并发、长期稳定性或性能测试。
- 上次失联是否由 Profile 并行解析引起仍未证实；本次显存竞争的直接根因则由日志、进程归属和释放显存后启动成功共同支持。
- 未访问另一实验机，未操作其他服务或容器，未运行 NPU 实验或远端重解析，未删除 ascend-lab-runtime 下任何文件。本地仅更新本脱敏记录并提交，未推送。

2026-11-05 00:34:02 UTC（a2fix-3）：执行 `systemctl --user disable qwen36-image.service` 成功（退出码 0）；复核该单元 is-enabled=disabled、is-active=inactive，原服务 `llm-backend-primary.service` 仍为 active、18081/health=200；未启动或停止服务，未重启宿主机，未删除容器、镜像、单元文件或模型数据，未操作其他服务。
