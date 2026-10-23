#!/usr/bin/env bash
# 在a2以ychy运行；Git worktree 5322528和制品校验已准备。
set -euo pipefail
base=/home/ychy/services/qwen36
release=$base/c16-20261015/repo
audit=$base/kv-ab-20261015
unit=/home/ychy/.config/systemd/user/qwen36-a2.service
python3 - "$base" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
new=json.loads((base/'kv-ab-20261015/c16-managed-config.json').read_text())
old=json.loads((base/'deployment.json').read_text())['config_and_artifacts']
assert new['effective_config']['QWEN36_C8_ENABLED']=='0'
for key,value in old['artifacts'].items():
    assert value['sha256']==new['artifacts'][key]['sha256'],key
PY
[[ -e "$audit/qwen36-a2.before-c16.service" ]] || cp "$unit" "$audit/qwen36-a2.before-c16.service"
[[ -e "$audit/deployment.before-c16.json" ]] || cp "$base/deployment.json" "$audit/deployment.before-c16.json"
python3 - "$release" "$audit" <<'PY'
from pathlib import Path
import sys
release,audit=map(Path,sys.argv[1:])
s=(release/'deploy/qwen36-a2.service').read_text().replace('/home/ychy/services/qwen36/repo',str(release))
(audit/'qwen36-a2.c16.service').write_text(s)
PY
systemctl --user stop qwen36-a2.service
podman stop -t 90 qwen36-c16-kv-ab-20261015
cp "$audit/qwen36-a2.c16.service" "$unit"
systemctl --user daemon-reload
systemctl --user start qwen36-a2.service
systemctl --user status qwen36-a2.service --no-pager -l
