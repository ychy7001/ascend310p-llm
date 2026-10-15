# clt 部署交接包

本目录是2026-10-15交接时快照。clt本机副本：`/home/clt/qwen36-handoff-20261015/`。

先阅读[部署指南](DEPLOYMENT.md)，再阅读[实测与选型报告](REPORT.md)。同事直接登录`clt@192.168.29.49`即可按指南操作，无需经a2复制模型。

| 文件 | 用途 |
|---|---|
| DEPLOYMENT.md | 路径清单、已有服务启停、API、测速、独立容器创建、C8回滚、排障 |
| REPORT.md | 适配方法、C16/C8实测、内存、默认参数依据、验收边界 |
| CAPACITY-ESTIMATES.md | 收紧缓存后的分档估算；尚未按最小页数验收 |
| current.env | 交接时参数快照，实际入口读取部署根目录candidate.env |
| start-current.sh | 交接时启动器快照，实际使用部署根目录start-candidate.sh |
| runtime-identity.json | 当前服务身份，模型名c8仅为API兼容 |
| VERIFIED-HOST.json | 实际health、NPU与容器/挂载核对记录 |
| HOST-RESOURCES.sha256 | clt实际运行源码、参数、算子及模型config/index的校验 |
| PACKAGE.sha256 | 本交接包文件校验 |

快速定位当前服务：

```bash
cat /home/clt/qwen36-optimized-20261015/current-default.json
curl --fail http://127.0.0.1:18016/health
```

校验clt现有资源（不修改服务）：

```bash
cd /home/clt/qwen36-handoff-20261015
sha256sum -c PACKAGE.sha256
sha256sum -c HOST-RESOURCES.sha256
```

HOST-RESOURCES包含366项运行资源；模型仅记录config与index，不包含全部大体积权重的重新哈希。完整59文件权重校验属于初次部署记录，本次未重新遍历全部权重。后续有意修改文件会使校验不同，应记录新版本，不能为通过校验覆盖现有更改。

运行时核心基于e3fc3f9e85d7cd372d8a225317f24abea6f67861，额外clt补丁与制品通过HOST-RESOURCES追溯；服务默认切换记录为9eb87f8。交接包不是可直接替换运行目录的安装包，不包含权重、镜像或算子二进制。主机本来已具备这些资源，见部署指南。
