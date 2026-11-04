#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${VLLM_ASCEND_MANIFEST_PATH:-/workspace/backend-primary.manifest.json}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[vllm-ascend-quadlet] missing manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

readarray -t MANIFEST_EXPORTS < <(
  python3 - "${MANIFEST_PATH}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))

env = payload.get("env", {}) or {}
for key, value in sorted(env.items()):
    if value is None:
        continue
    print(f"export {key}={shlex.quote(str(value))}")

command = payload.get("command", [])
if not isinstance(command, list) or not command:
    raise SystemExit("manifest missing command")

quoted = " ".join(shlex.quote(str(item)) for item in command)
print(f"COMMAND={quoted}")
PY
)

COMMAND=""
for line in "${MANIFEST_EXPORTS[@]}"; do
  if [[ "${line}" == COMMAND=* ]]; then
    COMMAND="${line#COMMAND=}"
  elif [[ -n "${line}" ]]; then
    eval "${line}"
  fi
done

if [[ -z "${COMMAND}" ]]; then
  echo "[vllm-ascend-quadlet] empty command in manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

# The validated 310P x86 image provides CANN's set_env.sh. Loading it here
# keeps the generated manifest free from host-version-specific library paths.
# Vendor environment scripts reference optional shell variables.
set +u
ASCEND_ENV_SCRIPT="/usr/local/Ascend/ascend-toolkit/set_env.sh"
if [[ -f "${ASCEND_ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_ENV_SCRIPT}"
elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# Export after the login profile, before importing the experimental worker.
PAIR_EXPORTS='export ASCEND_CUSTOM_OPP_PATH="/gdn-pair/opp/vendors/hyperqwen_gdn_pair921_transformer:/native-wy/opp/vendors/hyperqwen_wy921_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"; export PYTHONPATH="/controltools:/gdn-pair/tools:${PYTHONPATH:-}";'
exec bash -lc "${PAIR_EXPORTS} exec ${COMMAND}"
