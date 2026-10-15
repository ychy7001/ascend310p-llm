#!/usr/bin/env bash
set -euo pipefail
cd /home/clt/qwen36-optimized-20261015
source ./candidate.env
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    exec docker start "$CONTAINER_NAME"
fi
exec bash repo/experimental/run-c8-service.sh
