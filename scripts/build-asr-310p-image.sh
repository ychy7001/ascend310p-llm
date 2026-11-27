#!/usr/bin/env bash
set -euo pipefail

[[ $# == 1 ]] || { printf 'Usage: %s IMAGE_TAG\n' "$0" >&2; exit 2; }
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
base=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["image"])' \
    "$repo/deploy/image/asr-310p.lock.json")
grep -qxF "FROM $base AS base" "$repo/deploy/image/Containerfile.asr-310p" ||
    { printf 'Containerfile 的 base 与锁文件不一致，期望 %s\n' "$base" >&2; exit 1; }
podman image exists "$base" ||
    { printf '本机缺少 pinned base：%s\n' "$base" >&2; exit 1; }
exec podman build --pull=never --network=none --no-cache --arch=amd64 \
    -f "$repo/deploy/image/Containerfile.asr-310p" -t "$1" "$repo/deploy/image"
