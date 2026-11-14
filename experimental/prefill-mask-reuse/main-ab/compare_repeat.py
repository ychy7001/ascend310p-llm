"""Distinguish mixed-output drift reproduced by unpatched repeat from unmatched changes."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = [root/d for d in ('forced-12288-a', 'forced-12288-b', 'forced-repeat-12288-a')]
rows = []
for arm in arms:
    status = json.loads((arm/'state.json').read_text())
    assert status['ExitCode'] == 0 and not status['OOMKilled']
    log = (arm/'server.log').read_text()
    assert 'selected_compressed=False' in log and 'selected_compressed=True' not in log
    if arm != arms[1]:
        assert '[main-ab-mask] shape=(1, 768,' in log
    rows.append(json.loads((arm/'mixed.json').read_text()))
assert [(arm/'runtime-cli.txt').read_text() for arm in arms].count((arms[0]/'runtime-cli.txt').read_text()) == 3
result = []
for a, b, repeat in zip(*rows, strict=True):
    for k in ('slot', 'prompt_sha', 'cached', 'shape'):
        assert a[k] == b[k] == repeat[k], (k,a[k],b[k],repeat[k])
    result.append({'slot':a['slot'], 'cached':a['cached'], 'a_eq_b':a['text']==b['text'],
                   'a_eq_a_repeat':a['text']==repeat['text'], 'b_eq_a_repeat':b['text']==repeat['text']})
assert len(result)==4
print(json.dumps(result, indent=2))
