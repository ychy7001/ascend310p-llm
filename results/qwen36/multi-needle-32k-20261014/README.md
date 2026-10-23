# 32K 多位置 needle 检索（2026-10-14）

为避免单一 112K 位置 canary 的偶然性，在 `prompt_tokens=32256`、`max_tokens=14` 下分别将 `ASCEND732941` 放在 25%、50%、75%、90% 四个位置；MindIEQ 服务 `18024` 和原始 W4 服务 `18028` 使用相同脚本各执行一次。

| 位置 | MindIEQ | 原始 W4 |
|---:|---|---|
| 25% | PASS | PASS |
| 50% | PASS | PASS |
| 75% | PASS | PASS |
| 90% | PASS | PASS |

八个请求均返回 `ASCEND732941`，usage 均为 `32256 prompt + 14 completion`，finish reason 为 `stop`。这是多位置检索门禁，不是完整任务质量评测；112K 仍保留单位置 needle 结果作为长上下文容量证据。
