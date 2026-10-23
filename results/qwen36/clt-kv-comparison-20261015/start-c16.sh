#!/usr/bin/env bash
set -euo pipefail
source /home/clt/qwen36-kv-comparison-20261015/c16.env
exec bash /home/clt/qwen36-optimized-20261015/repo/experimental/run-c8-service.sh
