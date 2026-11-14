#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
block=${1:-64}
[[ $block =~ ^[0-9]+$ ]]
prefix=${NUMERIC_PREFIX:-numeric-$block}
probe=${NUMERIC_PROBE:-attention_numeric.py}
image=${NUMERIC_IMAGE:-fdbafc8bb69f5d2c6fbb39d461bb533d31a302fcaebfa6b264b6bc336a6fa56f}
[[ ! -e $prefix.log && ! -e $prefix-exit.txt ]] || { echo "Refusing to overwrite $prefix" >&2; exit 2; }
exec 9<>/tmp/a310-npu.lock
flock -w 900 9
[[ -z $(podman ps -q) && -z $(sudo -n podman ps -q) ]]
npu-smi info > "$prefix-before.txt"
grep -q 'No running processes found' "$prefix-before.txt"
cmd=(podman run --rm --pull=never --network=none --group-add keep-groups --ipc shareable --shm-size 8g --entrypoint bash)
for device in davinci0 davinci1 davinci_manager devmm_svm hisi_hdc; do cmd+=(--device /dev/"$device"); done
for file in /usr/local/bin/npu-smi /etc/ascend_install.info /usr/local/Ascend/driver/version.info /usr/local/Ascend/driver/lib64 /usr/local/dcmi; do cmd+=(-v "$file:$file:ro"); done
cmd+=(-v "$PWD:/bench:ro" "$image" -c
  'source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || exit 2
   export TORCH_DEVICE_BACKEND_AUTOLOAD=0 ATEN_CPU_CAPABILITY=default OMP_NUM_THREADS=1
   run_probe() {
     if [[ $2 == repro_future_mask.py ]]; then
       python3 /bench/"$2" --device "$3"
     else
       python3 /bench/"$2" "$3" "$1"
     fi
   }
   run_probe "$1" "$2" 0; rc0=$?
   run_probe "$1" "$2" 1; rc1=$?
   ((rc0 == 0 && rc1 == 0))' _ "$block" "$probe")
printf '%q ' "${cmd[@]}" > "$prefix-launch.txt"
set +e
"${cmd[@]}" > "$prefix.log" 2>&1
rc=$?
set -e
npu-smi info > "$prefix-final.txt"
grep -q 'No running processes found' "$prefix-final.txt" || rc=1
printf '%s\n' "$rc" > "$prefix-exit.txt"
exit "$rc"
