# WY registry交付记录（2026-11-08）

- TP1 W4A8：复用既有8e17ec9a…digest、long-prefill/MTP2 profile，不重建或扩展W4矩阵。
- TP2 W8A8：推送已验收的9df3c6f3…image ID，tag为`qwen36-tp2-w8a8-wy-20261108`，registry digest为e96595cb…；本变体MTP关闭。
- `registry-images.json`登记完整引用、config digest、rootfs diff IDs和核验时间；两个`*-remote-manifest.json`保留registry原始manifest字节。
- Registry V2 GET独立验证manifest SHA256、Docker-Content-Digest及config blob SHA256。部署不依赖tag后续是否变化。
- 原始TP1实验freeze.json与远端config均为b381d8e5…，此前文档image ID误写为86e64c35…已修正；镜像digest及实际制品未变。
- 配置入口为`deploy/w4a8-tp1/qwen36-image.env.example`和`deploy/w8a8-tp2-wy/qwen36-wy-image.env.example`。仅按宿主修改MODEL_DIR、端口与驱动设备映射，不在配置中存放registry凭据。
- 推送未重建运行时；TP2验收证据沿用同image ID的`docs/qwen36/wy-release-candidate.md`及其结果目录。本次未切换a1，也未覆盖stable tag。
- MTP扩展以`docs/qwen36/wy-mtp-integration-20261108.md`单独记录，不由本次registry发布推定通过。
- 另已通过podman按完整registry digest拉取W8，返回同一image ID；见`pull-w8.txt`。`final-cleanup.json`确认最终NPU空闲、无实验容器，任务临时凭据已删除。
