#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018
model=/home/ychy/models/Qwen3.8-Flash-Next-W4A16-G128-300i-d32978c
# Refuse to start a partial checkpoint or a failed GDN candidate.
python3 - "$work" "$model" <<'PY'
import json, sys
from pathlib import Path
work,model=map(Path,sys.argv[1:])
d=json.loads((model/'download-verified.json').read_text())
assert d['revision']=='d32978ccdde1729caec0034571b03a14779698e9'
assert len(d['files'])==1626
for rank in (0,1):
 r=json.loads((work/f'gdn-tp2-rank{rank}.json').read_text())
 assert r['hccl_sum']==3 and len(r['cases'])==10
 assert all(c['output_pass'] and c['state_max_abs_error']<=.002 for c in r['cases'])
 mask=json.loads((work/f'chunk-causal-rank{rank}.json').read_text())
 assert all(c['bad_elements']==0 for c in mask['cases'])
print('CHECKPOINT_AND_GDN_GATES_OK')
PY
systemctl --user stop qwen36-image.service
if podman container exists qwen36-tp1-chip1-w4-c8-original-112k; then
 podman stop --time 15 qwen36-tp1-chip1-w4-c8-original-112k
fi
podman run -d --name qwen38-tp2-text-4k-graph --network host --group-add keep-groups --shm-size 16g \
 --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v "$work:/experiment:rw" -v "$model:/model:ro" \
 -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
 -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e SOC_VERSION=ascend310p3 \
 localhost/qwen38-runtime:ffa980c-3ab5dda-20261018 bash /experiment/serve-text-tp2-graph.sh
podman inspect qwen38-tp2-text-4k-graph > "$work/qwen38-graph-serving-inspect.json"
