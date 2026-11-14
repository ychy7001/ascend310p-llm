#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
out=$root/${1:?Provide a fresh result directory name}
mkdir "$out"
profile=${2:-graph-mtp}
limit=${3:-262144}
port=18153
url=http://127.0.0.1:$port
name=qwen36-wy-compressed-screen
image=fdbafc8bb69f5d2c6fbb39d461bb533d31a302fcaebfa6b264b6bc336a6fa56f
model_dir=/srv/meetai/models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8
exec 9<>/tmp/a310-npu.lock
flock -n 9
idle() {
  [[ -z $(podman ps -q) && -z $(sudo -n podman ps -q) && -z $(ss -H -lnt "sport = :$port") ]] || return 1
  npu-smi info > "$1" && grep -q 'No running processes found' "$1"
}
idle "$out/npu-before.txt"
started=0
cleanup() {
  rc=$?
  trap - EXIT
  if ((started)); then
    podman stop --time 40 "$name" > "$out/stop.txt" 2>&1 || rc=1
    podman logs "$name" > "$out/server.log" 2>&1 || rc=1
    podman inspect "$name" --format '{{json .State}}' > "$out/state.json" || rc=1
    python3 - "$out/state.json" <<'PYCODE' || rc=1
import json, sys
state = json.load(open(sys.argv[1]))
assert state['ExitCode'] == 0 and not state['OOMKilled'], state
PYCODE
    podman rm "$name" >> "$out/stop.txt" 2>&1 || rc=1
  fi
  idle "$out/npu-final.txt" || rc=1
  printf '%s\n' "$rc" > "$out/exit.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
mkdir "$out/cache"
cmd=(podman run -d --pull=never --name "$name" --network host --group-add keep-groups --ipc shareable --shm-size 32g --entrypoint /bench/launch.sh)
for device in davinci0 davinci1 davinci_manager devmm_svm hisi_hdc; do cmd+=(--device /dev/"$device"); done
for file in /usr/local/bin/npu-smi /etc/ascend_install.info /usr/local/Ascend/driver/version.info /usr/local/Ascend/driver/lib64 /usr/local/dcmi; do cmd+=(-v "$file:$file:ro"); done
pkg=/vllm-workspace/vllm-ascend/vllm_ascend/_310p
cmd+=(-v "$model_dir:/model:ro" -v "$out/cache:/root/.cache:rw"
      -v "$root/launch.sh:/bench/launch.sh:ro"
      -v "$root/overlay/chunk_gated_delta_rule.py:$pkg/ops/fla/chunk_gated_delta_rule.py:ro"
      -v "$root/overlay/attention_v1.py:$pkg/attention/attention_v1.py:ro"
      -e PORT="$port" -e LIMIT="$limit" -e WY_PROFILE="$profile" -e GRAPH_SIZES="${GRAPH_SIZES:-3,6,12}" "$image")
printf '%q ' "${cmd[@]}" > "$out/launch.txt"
"${cmd[@]}" > "$out/cid"
started=1
for ((i=0;i<144;++i)); do
  if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then break; fi
  [[ $(podman inspect "$name" --format '{{.State.Running}}') == true ]]
  sleep 5
done
curl -fsS --max-time 5 "$url/v1/models" > "$out/models.json"
podman exec "$name" cat /proc/1/cmdline | tr '\0' '\n' > "$out/runtime-cli.txt"
curl -fsS --max-time 30 -H 'Content-Type: application/json' -d '{"model":"Qwen3.6-35B-A3B-w8a8","prompt":"Reply with the word OK.","max_tokens":16,"temperature":0}' "$url/v1/completions" > "$out/short.json"
timeout 150 python3 check_long_context_needle.py --base-url "$url" --model Qwen3.6-35B-A3B-w8a8 --out "$out/needle.jsonl"
curl -fsS --max-time 5 "$url/metrics" > "$out/metrics.txt"
podman logs "$name" > "$out/live.log" 2>&1
python3 - "$out" "$profile" <<'PY'
import json,re,sys
from pathlib import Path
out=Path(sys.argv[1]); log=(out/'live.log').read_text()
n=json.loads((out/'needle.jsonl').read_text().splitlines()[-1]); assert n['retrieved']
for route in ('native WY invoked','compressed SplitFuse invoked'):
    message='[wy-compressed-probe] '+route
    assert all(any(f'Worker_TP{rank}' in line and message in line for line in log.splitlines()) for rank in (0,1)), message
if sys.argv[2]=='graph-mtp':
    assert re.search(r'Replaying aclgraph|graph replay is active', log, re.I), 'Graph replay not seen'
    metrics=(out/'metrics.txt').read_text()
    assert any(float(line.split()[-1])>0 for line in metrics.splitlines()
               if re.match(r'vllm:spec_decode_num_drafts(_total)?[{ ]',line)), 'No MTP drafts'
PY
