"""Normalize an inspected bash command for run.py's explicit bash entrypoint."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('original', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
meta = json.loads(args.original.read_text())
cmd = meta[0]['Config']['Cmd']
if cmd[:2] != ['bash', '-lc']:
    raise SystemExit('Expected original command bash -lc; unknown entrypoint rejected')
meta[0]['Config']['Cmd'] = cmd[1:]
with args.output.open('x') as out:
    json.dump(meta, out, indent=2)
