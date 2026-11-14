#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9<>/tmp/a310-npu.lock
flock -n 9
base=$PWD
parent=$PWD/..
port=18152
mode=${1:-natural}
limit=${2:-12288}
[[ $mode == natural || $mode == forced || $mode == forced-repeat || $mode == graph-a || $mode == graph-b || $mode == mtp-a || $mode == mtp-b ]]
arms=(a b)
graph=0
case "$mode" in
  forced-repeat) arms=(a);;
  graph-*) arms=("${mode#graph-}"); graph=1;;
  mtp-*) arms=("${mode#mtp-}"); graph=2;;
esac
image=$(awk '{print $1}' "$parent/image-id.txt")
model_dir=$HOME/ascend-lab-runtime/hyperqwen-20261103/production-w8a8/model
sha256sum -c frozen-model.sha256
python3 prepare.py "$parent/source" "$base" > overlay.sha256
model=qwen36-main
endpoint=http://127.0.0.1:$port
name=
out=
sampler=
legacy_pair=0
idle() {
  local rootless rootful listeners
  rootless=$(podman ps -q) && rootful=$(sudo -n podman ps -q) && listeners=$(ss -H -lnt "sport = :$port") || return 1
  [[ -z $rootless && -z $rootful && -z $listeners ]] || return 1
  npu-smi info > "$1" || return 1
  grep -q 'No running processes found' "$1"
}
stop_arm() {
  if [[ -n $sampler ]]; then kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; sampler=; fi
  if [[ -n $name ]] && podman container exists "$name"; then
    if [[ $(podman inspect "$name" --format '{{.State.Running}}') == true ]]; then podman stop --time 40 "$name" > "$out/stop.txt"; fi
    podman logs "$name" > "$out/server.log" 2>&1
    podman inspect "$name" --format '{{json .State}}' > "$out/state.json"
    podman rm "$name" >> "$out/stop.txt"
  fi
  name=
}
cleanup() {
  rc=$?
  set +e
  stop_arm || rc=1
  idle "$base/$mode-$limit-final.txt" || rc=1
  printf '%s\n' "$rc" > "$base/$mode-$limit-exit.txt"
  if [[ $rc == 0 && $legacy_pair == 1 ]]; then
    if python3 summarize.py "$base" "$mode" "$limit" > "$base/$mode-$limit-summary.tmp"; then
      mv "$base/$mode-$limit-summary.tmp" "$base/$mode-$limit-summary.json"
    else
      rc=1
      printf '%s\n' "$rc" > "$base/$mode-$limit-exit.txt"
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
for arm in "${arms[@]}"; do
  out=$base/$mode-$limit-$arm
  mkdir -p "$out/cache"
  idle "$out/npu-before.txt"
  name=legacy-mask-main-$mode-$arm
  force=1; if [[ $mode == natural ]]; then force=0; fi
  cmd=(podman run -d --pull=never --name "$name" --network host --group-add keep-groups --ipc shareable --shm-size 32g --entrypoint /bench/launch.sh)
  for dev in davinci0 davinci1 davinci_manager devmm_svm hisi_hdc; do cmd+=(--device /dev/"$dev"); done
  for file in /usr/local/bin/npu-smi /etc/ascend_install.info /usr/local/Ascend/driver/version.info /usr/local/Ascend/driver/lib64 /usr/local/dcmi; do cmd+=(-v "$file:$file:ro"); done
  path=vllm_ascend/_310p/attention
  for file in attention_v1.py attention_mask.py; do cmd+=(-v "$base/$arm/$path/$file:/vllm-workspace/vllm-ascend/$path/$file:ro"); done
  cmd+=(-v "$base/launch.sh:/bench/launch.sh:ro" -v "$model_dir:/model:ro" -v "$out/cache:/root/.cache:rw")
  cmd+=(-e PORT="$port" -e LIMIT="$limit" -e MAIN_AB_FORCE_LEGACY="$force" -e MAIN_AB_GRAPH="$graph" "$image")
  printf '%q ' "${cmd[@]}" > "$out/launch.txt"
  "${cmd[@]}" > "$out/cid"
  for ((i=0; i<144; ++i)); do
    if curl -fsS --max-time 2 "$endpoint/health" >/dev/null 2>&1; then break; fi
    [[ $(podman inspect "$name" --format '{{.State.Running}}') == true ]]
    sleep 5
  done
  curl -fsS --max-time 5 "$endpoint/v1/models" > "$out/models.json"
  podman exec "$name" cat /proc/1/cmdline | tr '\0' '\n' > "$out/runtime-cli.txt"
  npu-smi info > "$out/npu-ready.txt"
  (while true; do date -Is; npu-smi info; sleep 2; done) > "$out/npu-workload-samples.txt" &
  sampler=$!
  if [[ $graph != 0 ]]; then
    timeout 240 python3 trace_graph.py "$endpoint" "$model" "$out"
    stop_arm
    python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["ExitCode"]==0 and not x["OOMKilled"], x' "$out/state.json"
    if [[ $graph == 1 ]]; then grep -Ei "ACL graph replay is active for the V2 target model" "$out/server.log"; fi
    grep -q "state=SpecDecoding .*selected_compressed=False" "$out/server.log"
    grep -Eq "\[main-ab-mask\] shape=\(1, 768, [0-9]+, 16\).*state=SpecDecoding" "$out/server.log"
    continue
  fi
  for tokens in 4096 8192; do
    timeout 240 python3 bench_server_phases.py --base-url "$endpoint" --model "$model" --prompt-tokens "$tokens" --output-tokens 1 --repeats 3 --cache-salt-prefix "$mode-$arm-$tokens-$(date +%s)" --out "$out/cold-$tokens.jsonl"
  done
  timeout 240 python3 check_long_context_needle.py --base-url "$endpoint" --model "$model" --prompt-tokens 8192 --out "$out/needle.jsonl"
  python3 smoke_mix.py "$endpoint" "$model" "$out"
  npu-smi info > "$out/npu-after-workload.txt"
  stop_arm
  python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["ExitCode"]==0 and not x["OOMKilled"], x' "$out/state.json"
  idle "$out/npu-after.txt"
  if [[ $mode == natural ]] && grep -q 'selected_compressed=True' "$out/server.log"; then break; fi
  grep -q '\[main-ab-mask\]' "$out/server.log"
  if [[ $arm == b ]]; then legacy_pair=1; elif [[ ${#arms[@]} == 2 ]]; then sleep 60; fi
done
