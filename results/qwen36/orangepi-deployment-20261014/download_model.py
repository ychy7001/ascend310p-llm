from modelscope import snapshot_download

snapshot_download(
    "ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P",
    local_dir="/data/models/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P",
    max_workers=4,
)
