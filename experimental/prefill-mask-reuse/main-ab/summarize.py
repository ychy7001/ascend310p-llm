#!/usr/bin/env python3
"""Summarize only completed, dispatch-confirmed legacy A/B windows."""
import json
from pathlib import Path
import re
from statistics import median
import sys

base = Path(sys.argv[1])
mode = sys.argv[2]
limit = sys.argv[3]
status = (base/f'{mode}-{limit}-exit.txt').read_text().strip()
assert status == '0' or (sys.argv[4:] == ['--recover-aggregation'] and status == '1')
assert 'No running processes found' in (base/f'{mode}-{limit}-final.txt').read_text()

def jsonl(file):
    return [json.loads(line) for line in file.read_text().splitlines()]

arms = {}
for arm in 'ab':
    root = base/f'{mode}-{limit}-{arm}'
    state = json.loads((root/'state.json').read_text())
    assert state['ExitCode'] == 0 and not state['OOMKilled'] and not state['Running']
    log = (root/'server.log').read_text()
    masks = re.findall(r'\[main-ab-mask\] shape=\(([^)]+)\) query=(\d+) state=(\w+)', log)
    assert len(masks) >= 2 and 'selected_compressed=False' in log and 'selected_compressed=True' not in log
    assert any(state == 'ChunkedPrefill' for _, _, state in masks), masks
    shape = [list(map(int, dims.split(','))) for dims, _, _ in masks]
    if arm == 'a':
        assert all(s[1]*16 == int(limit) for s in shape), shape
    else:
        assert any(s[1]*16 < int(limit) for s in shape), shape
    cold = {}
    for n in ('4096','8192'):
        xs = [r for r in jsonl(root/f'cold-{n}.jsonl') if r['kind']=='request']
        assert len(xs)==3 and all(r['valid'] and r['prefix_cache_delta']['vllm:prefix_cache_hits_total']==0 for r in xs)
        cold[n] = [{k:r[k] for k in ('prompt_ids_sha256','prefill_tok_s')} | {'ttft_s':r['response']['ttft_s'],
                   'text':r['response']['text']} for r in xs]
    mixed = json.loads((root/'mixed.json').read_text())
    assert len(mixed) == 4 and all((r['cached'] > 0) == r['warm'] for r in mixed)
    needle = jsonl(root/'needle.jsonl')[-1]
    assert needle['retrieved']
    sample = (root/'npu-workload-samples.txt').read_text()
    memory = {}
    for device,used in re.findall(r'^\|\s+\d+\s+(\d+)\s*\|[^|\n]*\|\s+\S+\s+(\d+)\s*/\s*\d+\s*\|[ \t]*$', sample, re.M):
        memory.setdefault(device,[]).append(int(used))
    assert set(memory) == {'0','1'}
    arms[arm] = dict(cli=(root/'runtime-cli.txt').read_text().splitlines(), first_legacy_shapes=shape, legacy_states=sorted({state for _, _, state in masks}), cold=cold,
                     mixed=[{k:r[k] for k in ('slot','shape','cached','prompt_sha','text')} for r in mixed],
                     needle=needle, sampled_peak_npu_mb={d:max(v) for d,v in memory.items()})
a,b = arms['a'],arms['b']
assert a['cli'] == b['cli']
assert [r['prompt_sha'] for r in a['mixed']] == [r['prompt_sha'] for r in b['mixed']]
for n in a['cold']:
    assert [x['prompt_ids_sha256'] for x in a['cold'][n]] == [x['prompt_ids_sha256'] for x in b['cold'][n]]
medians = {n:{k:[median(r[k] for r in arms[x]['cold'][n]) for x in 'ab'] for k in ('ttft_s','prefill_tok_s')}
           for n in a['cold']}
parity = {
    'cold_text_all_equal': all([r['text'] for r in a['cold'][n]] == [r['text'] for r in b['cold'][n]] for n in a['cold']),
    'mixed_text_equal_count': sum(x['text'] == y['text'] for x,y in zip(a['mixed'], b['mixed'], strict=True)),
}
print(json.dumps(dict(output_parity=parity, legacy_mode=mode, max_model_len=int(limit), arms=arms, cold_medians_ab=medians), indent=2))
