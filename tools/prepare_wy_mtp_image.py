#!/usr/bin/env python3
"""Package the TP2 legacy-SplitFuse compatibility layer over a locked WY image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--base-image',required=True)
args=parser.parse_args()
if args.output.exists(): raise SystemExit('output already exists')
root=Path(__file__).resolve().parents[1]
files={
    'deploy/w8a8-tp2-wy-mtp/start-wy-mtp.sh':'start-wy-mtp.sh',
    'bootstrap/wy_sitecustomize.py':'wy-compat/sitecustomize.py',
    'runtime/wy_compat/graph_safe_splitfuse.py':'wy-compat/graph_safe_splitfuse.py',
    'experimental/direct_graph_310.py':'wy-compat/direct_graph_310.py',
}
manifest={'base_image':args.base_image,'variant':'tp2-w8a8-wy-mtp','compatibility':'eager production main/draft; optional legacy SplitFuse stable Graph mask for diagnostics','artifacts':{}}
for source,target in files.items():
    dest=args.output/target
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/source,dest)
    manifest['artifacts'][target]=hashlib.sha256(dest.read_bytes()).hexdigest()
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(args.output/'Containerfile').write_text('''ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY start-wy-mtp.sh /opt/qwen36/start-wy-mtp.sh
COPY wy-compat/ /opt/qwen36/wy-compat/
COPY manifest.json /opt/qwen36/wy-mtp-manifest.json
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0
ENTRYPOINT ["bash", "/opt/qwen36/start-wy-mtp.sh"]
''')
print(json.dumps(manifest,indent=2))
