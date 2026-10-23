#!/usr/bin/env bash
# CANN 9.1 当前 RC 包的 msprof 不接受旧 --export=on；离线导出已有数据。
set -euo pipefail
profile_dir=${1:?pass the collected PROF_* directory inside the container}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
parser=/usr/local/Ascend/cann-9.1.0/tools/profiler/profiler_tool/analysis/msprof/msprof.py
python3 "$parser" import -dir "$profile_dir"
python3 "$parser" export summary -dir "$profile_dir"
