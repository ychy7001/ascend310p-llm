#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for ((i=0;i<90;++i)); do
  if [[ -s forced-repeat-12288-exit.txt ]] && flock -n /tmp/a310-npu.lock -c true; then break; fi
  sleep 10
done
[[ $(cat forced-repeat-12288-exit.txt) == 0 ]]
python3 compare_repeat.py . > forced-repeat-comparison.tmp
mv forced-repeat-comparison.tmp forced-repeat-comparison.json
if ! bash run.sh graph-a 12288; then
  echo 'Unpatched Graph/MTP admission failed; inspect baseline before running candidate.'
  exit 1
fi
sleep 60
bash run.sh graph-b 12288
