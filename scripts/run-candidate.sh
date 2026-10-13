#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 已验证配置入口；SDK/制品由调用者提供，不安装、不替换权重。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
  echo 'Usage: scripts/run-candidate.sh [--check]' >&2; exit 2
fi
# 先加载固定候选；仅prefix与实例标识允许显式覆盖。
prefix=${ENABLE_PREFIX_CACHING:-0}
port=${PORT:-18024}
name=${CONTAINER_NAME:-qwen36-tp1-48g-112k}
source "$root/configs/tp1-48g-112k.env"
export ENABLE_PREFIX_CACHING="$prefix" PORT="$port" CONTAINER_NAME="$name"
[[ "$prefix" == 0 || "$prefix" == 1 ]] || { echo "ENABLE_PREFIX_CACHING must be 0 or 1" >&2; exit 2; }
: "${MODEL_DIR:?}" "${OPP_DIR:?}" "${EXTENSION:?}" "${C8_LIBRARY:?}"
: "${GDN_OPP_DIR:?}" "${GDN_LIBRARY:?}" "${GDN_SOURCE:?}"
for path in "$MODEL_DIR/model.safetensors.index.json" "$EXTENSION" "$C8_LIBRARY" "$GDN_LIBRARY" "$GDN_SOURCE"; do
  [[ -r "$path" ]] || { echo "Missing artifact: $path" >&2; exit 2; }
done
for path in "$OPP_DIR" "$GDN_OPP_DIR"; do
  [[ -d "$path" ]] || { echo "Missing OPP directory: $path" >&2; exit 2; }
done
python3 - "$root" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root=Path(sys.argv[1])
files={key:{'path':os.environ[key], 'sha256':hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()}
       for key in ('EXTENSION','C8_LIBRARY','GDN_LIBRARY','GDN_SOURCE')}
files['MODEL_INDEX']={'sha256':hashlib.sha256((Path(os.environ['MODEL_DIR'])/'model.safetensors.index.json').read_bytes()).hexdigest()}
print(json.dumps({'profile':(root/'configs/tp1-48g-112k.env').read_text(),'artifacts':files,
                  'prefix':os.environ['ENABLE_PREFIX_CACHING'],'port':os.environ['PORT']},ensure_ascii=False,indent=2))
PY
if [[ ${1:-} == --check ]]; then exit 0; fi
exec bash "$root/experimental/run-gdn-112k-product.sh"
