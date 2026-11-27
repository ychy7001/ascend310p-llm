#!/usr/bin/env bash
set -eo pipefail

if (( $# )); then
    # Explicit commands also work on hosts without the Ascend driver.
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ASCEND_HOME_PATH/x86_64-linux/devlib"
    exec "$@"
fi

: "${R2T2_WS_SECRET:?R2T2_WS_SECRET must be set}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
exec python3 /opt/confucius/ws_server.py \
    -ip "${R2T2_WS_LISTEN_ADDR:-0.0.0.0}" -p "${R2T2_WS_LISTEN_PORT:-18272}" \
    -m /model --vad_model_path /opt/confucius/checkpoints/vad/Stream-VAD
