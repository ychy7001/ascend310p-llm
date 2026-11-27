#!/usr/bin/env bash
set -euo pipefail

[[ $# == 1 ]] || { printf 'Usage: %s IMAGE_TAG\n' "$0" >&2; exit 2; }
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
exec podman build --pull=never --network=none --no-cache --arch=amd64 \
    -f "$repo/deploy/image/Containerfile.asr-310p" -t "$1" "$repo/deploy/image"
