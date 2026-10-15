#!/usr/bin/env bash
set -euo pipefail
cd /home/clt/qwen36-optimized-20261015
source ./candidate.env
exec bash repo/experimental/run-c8-service.sh
