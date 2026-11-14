#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9<>/tmp/a310-npu.lock
flock -n 9
model_dir=$HOME/ascend-lab-runtime/hyperqwen-20261103/production-w8a8/model
port=18150
name=qwen36-wy-packaged-mask-smoke
url=http://127.0.0.1:$port
image=$(python3 -c 'import json; print(json.load(open("image.json"))["image_id"])')
idle() {
  local rootless rootful listeners
  rootless=$(podman ps -q) && rootful=$(sudo -n podman ps -q) && listeners=$(ss -H -lnt "sport = :$port") || return 1
  [[ -z $rootless && -z $rootful && -z $listeners ]] || return 1
  npu-smi info > "$1" && grep -q 'No running processes found' "$1"
}
idle npu-before.txt
sha256sum -c frozen-model.sha256
mkdir -p cache
started=0
cleanup() {
  rc=$?
  trap - EXIT
  if ((started)); then
    podman stop --time 40 "$name" > stop.txt 2>&1 || rc=1
    podman logs "$name" > server.log 2>&1 || rc=1
    podman inspect "$name" --format '{{json .State}}' > state.json || rc=1
    podman rm "$name" >> stop.txt 2>&1 || rc=1
    python3 -c 'import json; x=json.load(open("state.json")); assert x["ExitCode"]==0 and not x["OOMKilled"]' || rc=1
    [[ $(grep -cF '[eager-mask-reuse] first hit' server.log) == 2 ]] || rc=1
    grep -qF 'Graph capturing finished' server.log || rc=1
    grep -qF 'Replaying aclgraph' server.log || rc=1
  fi
  idle npu-final.txt || rc=1
  printf '%s\n' "$rc" > smoke-exit.txt
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
cmd=(podman run -d --pull=never --name "$name" --network host --group-add keep-groups --ipc shareable --shm-size 32g)
for dev in davinci0 davinci1 davinci_manager devmm_svm hisi_hdc; do cmd+=(--device /dev/"$dev"); done
for file in /usr/local/bin/npu-smi /etc/ascend_install.info /usr/local/Ascend/driver/version.info /usr/local/Ascend/driver/lib64 /usr/local/dcmi; do cmd+=(-v "$file:$file:ro"); done
cmd+=(-v "$model_dir:/model:ro" -v "$PWD/cache:/root/.cache:rw")
cmd+=(-e HOST=127.0.0.1 -e PORT="$port" -e SERVED_MODEL_NAME=Qwen3.6-35B-A3B-w8a8 -e MAX_MODEL_LEN=262144 -e MAX_BATCHED_TOKENS=2048 -e MAX_NUM_SEQS=4 -e MEMORY_UTILIZATION=0.90 -e ENABLE_PREFIX_CACHING=1 -e ASYNC_SCHEDULING=1 -e NUM_SPECULATIVE_TOKENS=2 -e ENFORCE_EAGER=0 -e DRAFT_ENFORCE_EAGER=1 -e GRAPH_CAPTURE_SIZES=3,6,12 "$image")
printf '%q ' "${cmd[@]}" > launch.txt
"${cmd[@]}" > cid
started=1
for ((i=0;i<144;++i)); do
  if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then break; fi
  [[ $(podman inspect "$name" --format '{{.State.Running}}') == true ]]
  sleep 5
done
curl -fsS --max-time 5 "$url/v1/models" > models.json
podman exec "$name" cat /proc/1/cmdline | tr '\0' '\n' > runtime-cli.txt
timeout 150 python3 check_long_context_needle.py --base-url "$url" --model Qwen3.6-35B-A3B-w8a8 --out needle.jsonl
python3 -c 'import json; assert json.loads(open("needle.jsonl").read().splitlines()[-1])["retrieved"]'
