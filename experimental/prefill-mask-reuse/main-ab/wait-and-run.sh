#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for ((i=0; i<720; ++i)); do
  [[ ! -s ../image-id.txt ]] || break
  for status in recover-pinned-exit.txt build-release-exit.txt; do
    if [[ -s ../$status ]] && [[ $(cat "../$status") != 0 ]]; then
      printf 'Stopped before the NPU run: %s = %s\n' "$status" "$(cat "../$status")" >&2
      exit 1
    fi
  done
  sleep 20
done
if [[ ! -s ../image-id.txt ]]; then echo 'Timed out waiting for the validated main image' >&2; exit 124; fi
bash run.sh natural 12288
if grep -q 'selected_compressed=True' natural-12288-a/server.log; then
  sleep 60
  bash run.sh forced 12288
fi
